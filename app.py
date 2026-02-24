DEBUG_WEB_AGENT = True   # set False in prod
WEB_LOG_PREVIEW_CHARS = 100000

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import boto3
from botocore.config import Config
import botocore
import json
import logging
import asyncio
import time
import random
from concurrent.futures import ThreadPoolExecutor
import os

# 🔹 IMPORT WEBSITE SCRAPER
from scraper import get_url, get_data_xml

# =====================================================================
# LOGGING
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
log = logging.getLogger("APP")

# === AWS Bedrock Client (from env vars) ===

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "ap-south-1"),
    config=Config(
        read_timeout=600,
        connect_timeout=60,
        retries={"max_attempts": 3}
    )
)

# =====================================================================
# GLOBAL CONCURRENCY LIMITS (UNCHANGED)
# =====================================================================

COMPRESS_LIMIT = 3
FINAL_EXTRACT_LIMIT = 1

executor = ThreadPoolExecutor(max_workers=COMPRESS_LIMIT)

# =====================================================================
# FASTAPI APP
# =====================================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# HELPERS (UNCHANGED)
# =====================================================================

def chunk_text(text: str, max_chars: int, overlap: int = 3000):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)

        # Move start forward but keep overlap
        start = end - overlap

        # Safety check to avoid infinite loop
        if start < 0:
            start = 0

    return chunks


def clean_llm_output(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.partition("```")[2]
        t = t.partition("```")[0]
        t = t.replace("json", "", 1).strip()
    return t.strip("`").strip()

# =====================================================================
# 🧠 AGENT 1 — COMPANY IDENTIFIER
# =====================================================================

def identify_company_name(text: str) -> str:
    prompt = f"""
Extract the exact legal company name from the text.
Return ONLY the company name.

TEXT:
{text[:6000]}
"""

    response = bedrock.invoke_model(
        modelId="arn:aws:bedrock:ap-south-1:241533142399:inference-profile/global.anthropic.claude-opus-4-6-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0,
            "anthropic_version": "bedrock-2023-05-31"
        })
    )

    out = json.loads(response["body"].read())
    return out["content"][0]["text"].strip()

# =====================================================================
# 🌐 AGENT 2 — WEBSITE AGENT (INDEX ROUTED)
# =====================================================================

def extract_web_sections(company_name: str, schema: list, section_indexes: list[int]):
    url = get_url(company_name)
    log.info(f"[WEB] Scraping URL: {url}")

    web_text = get_data_xml(url)

    if not web_text or "Error" in web_text:
        log.warning("[WEB] Empty or error response from scraper")
        return {}

    # ---------------- LOGGING ----------------
    log.info(f"[WEB] Scraped content length: {len(web_text)} characters")

    if DEBUG_WEB_AGENT:
        preview = web_text[:WEB_LOG_PREVIEW_CHARS]
        log.info(
            "[WEB] Scraped content preview (first %d chars):\n%s",
            WEB_LOG_PREVIEW_CHARS,
            preview
        )
    # ----------------------------------------

    selected_schema = [schema[i] for i in section_indexes]

    prompt = f"""
You are an information extraction engine.

INSTRUCTIONS:
- You are IPO analyst who has good understanding on the IPO Terms.
- You understand IPO terminology, tables, and financial statements.
- You may internally reason about meanings, synonyms, and structure.
- DO NOT explain your reasoning.
- DO NOT include any text outside JSON.
- Issue price, Share Holding pre issue and share holding post issue should be captured properly. ENSURE you check and rerun to get right values.
- STRICTLY EXTACT ALL DETAILED OBJECTS OF THE ISSUE needs to extracted as Objective of the issue 1, Amount 1, Objective of the issue 2, Amount 2.

TASK:
Fill the given schema using ONLY information present in the text.
If a schema field conceptually matches information in the text, fill it.
If the information is not present, leave the value empty ("").

IMPORTANT OUTPUT RULES:
- Output ONLY valid JSON.
- No markdown.
- No explanations.
- No comments.
- No trailing commas.
Return ONLY valid JSON.
Ensure it is parseable by Python json.loads().
Double-check commas and quotes before finishing.

SCHEMA:
{json.dumps(selected_schema, ensure_ascii=False)}

TEXT:
{web_text}
"""

    response = bedrock.invoke_model(
        modelId="arn:aws:bedrock:ap-south-1:241533142399:inference-profile/global.anthropic.claude-opus-4-6-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8000,
            "temperature": 0,
            "anthropic_version": "bedrock-2023-05-31"
        })
    )

    raw = json.loads(response["body"].read())["content"][0]["text"]
    cleaned = clean_llm_output(raw)

    if not cleaned.strip():
        log.error("[WEB] LLM returned empty output")
        return {}

    try:
        extracted = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.error("[WEB] Invalid JSON from LLM at pos %s: %s", e.pos, e.msg)
        log.error("[WEB] Raw LLM output:\n%s", cleaned[:2000])
        return {}

    return {
        section_indexes[i]: extracted[i]
        for i in range(len(section_indexes))
    }

# =====================================================================
# 📄 AGENT 3 — PDF AGENT (100% ORIGINAL CODE)
# =====================================================================

def compress_chunk(chunk: str) -> str:
    request_body = {
        "schemaVersion": "messages-v1",
        "system": [{
            "text": (
                "You are a STRICT structured information extraction engine.\n"
                "You are NOT a summarizer.\n"
                "You must EXTRACT and PRESERVE factual information exactly as written.\n\n"

                "GENERAL RULES:\n"
                "1. Extract ONLY content that appears in this chunk.\n"
                "2. DO NOT summarize.\n"
                "3. DO NOT compress or shorten numeric values.\n"
                "4. Preserve ALL numbers, percentages, rupee values exactly.\n"
                "5. If a section does NOT appear in this chunk, DO NOT mention it.\n"
                "6. DO NOT write 'Couldn't be captured'.\n"
                "7. DO NOT invent data.\n"
                "8. DO NOT explain anything.\n\n"

                "TABLE EXTRACTION RULES (VERY IMPORTANT):\n"
                "1. If a financial table appears, you MUST extract the COMPLETE table.\n"
                "2. Preserve ALL rows exactly as shown.\n"
                "3. Preserve ALL columns exactly as shown.\n"
                "4. If headers span multiple lines, intelligently combine them into a single column name.\n"
                "5. DO NOT drop any peer rows.\n"
                "6. DO NOT merge numeric columns.\n"
                "7. If only part of a table appears in this chunk, extract only the visible rows.\n"
                "8. Maintain row-wise structure in JSON array format.\n\n"

                "For the section 'Comparison of Accounting Ratios of Listed Peer Companies',\n"
                "return it STRICTLY in this JSON structure:\n\n"

                "{\n"
                "  \"Comparison of Accounting Ratios of Listed Peer Companies\": [\n"
                "    {\n"
                "      \"Company_Name\": \"\",\n"
                "      \"Face_Value\": \"\",\n"
                "      \"Total_Revenue_FY2025\": \"\",\n"
                "      \"EPS_Basic_FY2025\": \"\",\n"
                "      \"EPS_Diluted_FY2025\": \"\",\n"
                "      \"NAV_per_Equity_Share\": \"\",\n"
                "      \"PE_Based_on_Diluted_EPS\": \"\",\n"
                "      \"RONW_Percent\": \"\"\n"
                "    }\n"
                "  ]\n"
                "}\n\n"

                "Only include keys for sections actually present in this chunk.\n\n"

                "Possible section names include:\n"
                "Company Details\n"
                "Background\n"
                "DRHP & Approvals\n"
                "Listing & Intermediaries\n"
                "IPO Details\n"
                "Shares Offered\n"
                "Financials FY23–FY25\n"
                "Projections\n"
                "Shareholding Pattern\n"
                "Compliance\n"
                "Industry\n"
                "Objects\n"
                "Valuation\n"
                "Comparison of Accounting Ratios of Listed Peer Companies\n"
                "Lots & Shares\n"
                "Qualitative Factors\n"
                "Track Record\n"
                "Risk & Compliance\n"
                "Financial Statement Restated\n"
                "Other Regulatory and Statutory Approvals\n"
                "Price Information of Past Issues Handled\n"
                "SHARE HOLDER PATTERN\n"
                "Government and Statutory Approvals\n"
            )
        }]

            ,
        "messages": [{
            "role": "user",
            "content": [{
                "text": (
                    "Extract relevant IPO-related content from the following text chunk.\n\n"
                    "TEXT CHUNK:\n\n"
                    f"{chunk}"
                )
            }]
        }]
        ,
        "inferenceConfig": {
            "maxTokens": 1200,
            "temperature": 0.1,
            "topP": 0.8,
            "topK": 50
        }
    }

    retries = 8
    for attempt in range(retries):
        try:
            response = bedrock.invoke_model(
                modelId="arn:aws:bedrock:ap-south-1:241533142399:inference-profile/global.amazon.nova-2-lite-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            result = json.loads(response["body"].read())
            return result["output"]["message"]["content"][0]["text"]

        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                wait = (2 ** attempt) + random.random()
                log.warning(f"Nova throttled. Retry {attempt+1} in {wait:.2f}s")
                time.sleep(wait)
            else:
                raise

    raise Exception("Compression failed after retries")


# -------------------- SCHEMA FILTERING (UNCHANGED) --------------------

def filter_to_schema(llm_output, schema):
    cleaned = []

    if not isinstance(llm_output, list):
        llm_output = [llm_output]

    for index, section in enumerate(schema):
        llm_section = llm_output[index] if index < len(llm_output) else {}
        new_sec = {}

        for key, value in section.items():

            # If schema expects a dictionary
            if isinstance(value, dict):
                new_sec[key] = llm_section.get(key, {})

            # If schema expects a list (like peer comparison)
            elif isinstance(value, list):
                new_sec[key] = llm_section.get(key, [])

            # Otherwise scalar
            else:
                new_sec[key] = llm_section.get(key, "")

        cleaned.append(new_sec)

    return cleaned


def extract_from_summary(summary, schema):
    prompt = f"""
    You are a Anchor investment screening committee assistant
You are a STRICT JSON extraction engine.
Output EXACT JSON matching this schema.
Ensure that keys like Address_of_Operating__facilities include ALL facility addresses.
ENSURE for field in json "Commencement_of_Operations" you either give starting date or year of the company/business/etc.
ENSURE "MSME_status/_Udyam_Status" is filled using Government and statutory approvals section. If data not available just give if it looks like MSME or UDYAM. NEVER Respond "MSME_status/_Udyam_Status" with Couldn't be captured for this field.
ENSURE WITHOUT MISS For the below fields in json, answer with Yes or No or Not available with one liner reason.NEVER Say Couldn't be captured for the below fields.
      "DRHP_should_have_been_approved_by_any_Stock_Exchange_/_SEBI.": "",
      "Should_have_availed_at_least_one_Debt_products_from_Banks/FIs_and_servicing_it_for_at_least_3_years_prior_to_the_date_of_filing_of_DRHP.__\nand_/_or_\nshould_have_availed_funding_from_any_SEBI_registered_AIFs_at_least_1_year_prior_to_the_date_of_filing_of_DRHP.": "",
      "Net_worth_>_₹_20_crore": "",
      "EBIDTA_margin__-_₹2.00_cr._or_>_5%_(whichever_is_more)_for_at_least_two_out_of_the_three_most_recent_financial_years": "",
      "Total_Income__>_₹50_crore": "",
      "Net_Tangible_Assets_>_₹_2.50_crore": "",
      "Profitability_(total_PBT_of_3_years)_>_₹_10_crore": "",
      "Exposure_cap_per_MSME:_\nMinimum:_₹_1_crore_\nMaximum:_Up_to_₹_20_crore._\nNote:__Anchor_investment_shall_not_exceed 50%_of_the_Anchor_portion_of_the_Issue_or_10%_of_the_post_issue_paid-up_capital_of_the_issuer_company,_whichever_is_lower.": "",
      "Minimum_Promoters_Shareholding_\n(Post_Issue)-_60%\nIn_case_where_the_Company_has_received_any_investment_from_a_SEBI_registered_AIF,_the_Investment_Committee_may_take_a_suitable_view_on_the_minimum_level_of_Promoter_Shareholding_below_60%_since_the_AIF_would_want_to_partially_/_fully_exit_in_the_proposed_IPO_/_Offer_for_Sale_(OFS).": "",
      "Complied_with_Minimum_₹_2_Cr._Bid_criteria_of_SEBI": "",
      "Complied_with_RBI's_guideline_limiting_post-issue_shareholding_exposure_to_a_maximum_of_10%.": ""

Ensure you re-run once or fetch missed values.

IMPORTANT NOTE:
1. ALL CURRENCY ONLY IN INDIAN FORMATS.
2. REPLACE ANY  [●]  or blanks WITH "Couldn't be captured"
3. Never mention "SEBI Reg. No" anywhere in the values

NOTE:
for JSON field "About_Lead_and_Co-lead_merchant_bankers,_their_past_track_record"
Need to be calculated as below:
1. Give brief of 'Other regulatory and statutory disclosures' section.
2. Give brief of 'Price information of past issues handled'.
3. Calculate the percentage of issues trading in positive/Negative on their 180th day of listing from the table given there.

NOTE: ALWAYS REPLACE  [●] or 'NA' with 'Couldn't be captured'

SCHEMA:
{json.dumps(schema, ensure_ascii=False)}

TEXT:
{summary}
"""

    retries = 5
    for attempt in range(retries):
        try:
            response = bedrock.invoke_model(
                modelId="arn:aws:bedrock:ap-south-1:241533142399:inference-profile/global.anthropic.claude-opus-4-6-v1",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 30000,
                    "temperature": 0,
                    "anthropic_version": "bedrock-2023-05-31"
                })
            )

            raw = json.loads(response["body"].read())["content"][0]["text"]
            parsed = json.loads(clean_llm_output(raw))
            return filter_to_schema(parsed, schema)

        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                wait = (2 ** attempt) + random.random()
                log.warning(f"Claude throttled. Retry {attempt+1} in {wait:.2f}s")
                time.sleep(wait)
            else:
                raise

    raise Exception("Claude extraction failed")


# =====================================================================
# 🧩 STITCHER (INDEX BASED)
# =====================================================================

def stitch_by_index(pdf_json: list, web_json_by_index: dict):
    final = pdf_json.copy()
    for idx, section in web_json_by_index.items():
        final[idx] = section
    return final


# =====================================================================
# 🚀 ENDPOINT — FULL AGENTIC PIPELINE
# =====================================================================

@app.post("/extract")
async def extract_and_return_json(
    txt_file: UploadFile = File(...),
    input_schema: str = Form(...),
    web_sections: str = Form(""),
    chunk_size: int = Form(15000),
    semaphore_limit: int = Form(COMPRESS_LIMIT)
):
    start = time.time()

    text = (await txt_file.read()).decode("utf-8")
    schema = json.loads(input_schema)

    # 🧠 Agent 1
    company_name = identify_company_name(text)
    log.info(f"Company identified: {company_name}")

    # 📄 Agent 3 — PDF AGENT (UNCHANGED FLOW)
    chunks = chunk_text(text, chunk_size)
    log.info(f"Created {len(chunks)} chunks (chunk_size={chunk_size})")

    sem = asyncio.Semaphore(semaphore_limit)

    async def run_one(index, chunk):
        async with sem:
            await asyncio.sleep(0.2)

            t0 = time.time()
            out = await asyncio.get_event_loop().run_in_executor(
                executor,
                compress_chunk,
                chunk
            )
            log.info(f"Compressed chunk {index} in {time.time() - t0:.2f}s")
            return out

    tasks = [
        asyncio.create_task(run_one(i + 1, ch))
        for i, ch in enumerate(chunks)
    ]

    compressed = await asyncio.gather(*tasks)

    combined_summary = "\n\n".join(compressed)
    log.info(f"Combined summary length: {len(combined_summary)} chars")
    # log.info(print(combined_summary))

    pdf_json = await asyncio.get_event_loop().run_in_executor(
        ThreadPoolExecutor(max_workers=FINAL_EXTRACT_LIMIT),
        extract_from_summary,
        combined_summary,
        schema
    )



    # 🌐 Agent 2 — WEBSITE AGENT (SELECTIVE)
    web_json_by_index = {}
    if web_sections.strip():
        indexes = [int(x.strip()) - 1 for x in web_sections.split(",")]
        web_json_by_index = await asyncio.get_event_loop().run_in_executor(
            None,
            extract_web_sections,
            company_name,
            schema,
            indexes
        )

    # 🧩 STITCH
    final_json = stitch_by_index(pdf_json, web_json_by_index)

    log.info(f"TOTAL PIPELINE TIME: {time.time() - start:.2f}s")

    return {
        "company": company_name,
        "extracted_json": final_json
    }


# =====================================================================

@app.get("/")
def health():
    return {"status": "running"}


# =====================================================================

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9000)
