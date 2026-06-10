import requests

BASE = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_tests(tests):
    print("=" * 70)
    passed = 0
    failed = 0

    for test in tests:
        print(f"\nTest  : {test['name']}")
        try:
            if test["method"] == "GET":
                r = requests.get(test["url"], timeout=10)
            else:
                r = requests.post(test["url"], json=test["body"], timeout=10)

            print(f"Status: {r.status_code}")
            data = r.json()
            print(f"Result: {data}")

            expected = test.get("expected_label")
            if expected:
                got = data.get("label")
                if got == expected:
                    print(f"✅ PASS — expected '{expected}', got '{got}'")
                    passed += 1
                else:
                    print(f"❌ FAIL — expected '{expected}', got '{got}'")
                    failed += 1
            else:
                passed += 1  # No label expectation — just checking it runs

        except Exception as e:
            print(f"ERROR : {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. BASIC / INFRASTRUCTURE
# ---------------------------------------------------------------------------

basic_tests = [
    {
        "name": "Health Check",
        "method": "GET",
        "url": f"{BASE}/",
        "body": None,
    },
    {
        "name": "Empty text",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {"text": ""},
    },
    {
        "name": "Only spaces",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {"text": "     "},
    },
    {
        "name": "Too short text (< 10 words)",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {"text": "risk fraud loss"},
    },
    {
        "name": "Only numbers",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {"text": "123 456 789 100 200 300 400 500 600 700 800 900"},
    },
    {
        "name": "Only punctuation / symbols",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {"text": "!!! ??? ### $$$ @@@ &&& *** ... --- +++ ///"},
    },
    {
        "name": "Repeated single word",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {"text": "risk " * 50},
    },
]


# ---------------------------------------------------------------------------
# 2. UNSTRUCTURED — HIGH RISK
# ---------------------------------------------------------------------------

high_risk_tests = [
    {
        "name": "UNSTRUCTURED | High — bankruptcy + fraud + litigation",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "The company faces significant bankruptcy risk and ongoing litigation. "
                "We have reported material weakness in internal controls and negative cash flow. "
                "Potential covenant breach and regulatory fraud investigations have created "
                "severe uncertainty. Going concern doubts have been raised by our auditors."
            )
        },
        "expected_label": "high_risk",
    },
    {
        "name": "UNSTRUCTURED | High — cybersecurity + data breach",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "Our systems experienced multiple data breach incidents exposing customer data. "
                "Cybersecurity vulnerabilities have led to regulatory investigations and penalties. "
                "Material weakness in our security infrastructure creates ongoing liability exposure. "
                "Litigation from affected customers and shareholders represents significant adverse risk. "
                "The company faces potential delisting and severe financial impairment."
            )
        },
        "expected_label": "high_risk",
    },
    {
        "name": "UNSTRUCTURED | High — going concern + liquidity crisis",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "Our auditors have expressed going concern doubt about our ability to continue operations. "
                "Severe liquidity risk and negative cash flow threaten our ability to meet obligations. "
                "We have breached multiple debt covenants and face potential insolvency proceedings. "
                "Restructuring efforts have failed to address the underlying deficit and impairment losses. "
                "Regulatory sanctions and fraud investigations further complicate our recovery prospects."
            )
        },
        "expected_label": "high_risk",
    },
    {
        "name": "UNSTRUCTURED | High — macroeconomic headwinds",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "Inflation and recession fears have severely impacted our revenue projections. "
                "Volatile markets and supply chain disruption have increased operational costs significantly. "
                "Tariff increases and regulatory sanctions have disrupted our international business. "
                "Declining margins and increasing loss exposure threaten long-term viability. "
                "We face significant uncertainty about our ability to service existing debt obligations."
            )
        },
        "expected_label": "high_risk",
    },
]


# ---------------------------------------------------------------------------
# 3. UNSTRUCTURED — LOW RISK
# ---------------------------------------------------------------------------

low_risk_tests = [
    {
        "name": "UNSTRUCTURED | Low — record revenue + strong growth",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "The company reported record revenue and strong growth this quarter. "
                "Positive cash flow, no material litigation, and robust demand across all segments. "
                "We exceeded expectations with improved margins and a strong liquidity position. "
                "The company remains debt free with a stable and profitable outlook."
            )
        },
        "expected_label": "low_risk",
    },
    {
        "name": "UNSTRUCTURED | Low — market leader + innovation",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "As the market leader in our segment, we achieved record milestones this fiscal year. "
                "Innovative product launches drove revenue expansion and increased profitability. "
                "Strong liquidity, no material litigation, and efficient operations underpin our success. "
                "Dividend increases and share buybacks reflect our confidence in continued profitable growth. "
                "Customer satisfaction scores reached all-time highs with robust demand outlook."
            )
        },
        "expected_label": "low_risk",
    },
    {
        "name": "UNSTRUCTURED | Low — debt free + exceeded expectations",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "We are pleased to report another year of record revenue growth and expanded margins. "
                "The company is debt free with strong positive cash flow across all business units. "
                "No material litigation or regulatory issues impact our stable operations. "
                "Our innovative pipeline and market expansion strategy position us favorably. "
                "We exceeded analyst expectations and raised full-year guidance confidently."
            )
        },
        "expected_label": "low_risk",
    },
]


# ---------------------------------------------------------------------------
# 4. UNSTRUCTURED — MEDIUM RISK
# ---------------------------------------------------------------------------

medium_risk_tests = [
    {
        "name": "UNSTRUCTURED | Medium — mixed growth and challenges",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "The company achieved moderate revenue growth despite some market volatility. "
                "Stable operations and positive cash flow offset uncertainty in emerging markets. "
                "No material litigation exists, though regulatory compliance costs are rising. "
                "Inflation presents ongoing margin pressure but management has taken corrective action. "
                "We remain cautiously optimistic about the coming fiscal year with targeted investments."
            )
        },
        "expected_label": "medium_risk",
    },
    {
        "name": "UNSTRUCTURED | Medium — competitive pressure + stable core",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "Competitive pressure in our core markets has moderated revenue growth this year. "
                "While profitability remains positive, margins have compressed due to input cost inflation. "
                "We face uncertainty in two of our five business segments but maintain stable overall performance. "
                "No going concern issues exist and liquidity remains adequate for near-term obligations. "
                "Management is actively addressing operational challenges through restructuring initiatives."
            )
        },
        "expected_label": "medium_risk",
    },
]


# ---------------------------------------------------------------------------
# 5. STRUCTURED MODE — PROPER SECTION INPUT
# ---------------------------------------------------------------------------

risk_hi = (
    "The company faces bankruptcy risk, ongoing litigation, "
    "material weakness in internal controls, and fraud investigations."
)
mda_hi = (
    "Revenue declined 15% year over year. Negative cash flow and "
    "covenant breach risk remain significant concerns heading into next quarter."
)
biz_hi = (
    "We operate in highly competitive markets with significant disruption risk "
    "from new entrants and changing regulatory environment."
)
fin_hi = (
    "Net loss of $42M was recorded. Liquidity concerns persist with "
    "current ratio below 1.0 and going concern noted by auditors."
)

risk_lo = (
    "No material litigation or regulatory investigations exist. "
    "The company has strong internal controls with no identified weaknesses."
)
mda_lo = (
    "Revenue increased 22% year over year driven by strong demand. "
    "Positive cash flow and improved margins reflect operational excellence."
)
biz_lo = (
    "As the market leader we continue to expand into new geographies. "
    "Innovative products and strong brand loyalty drive sustainable growth."
)
fin_lo = (
    "Net income of $180M represents a record high. "
    "Strong liquidity with current ratio of 2.8 and zero long-term debt."
)

structured_tests = [
    {
        "name": "STRUCTURED | High risk filing",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": f"{risk_hi} {mda_hi} {biz_hi} {fin_hi}",
            "risk_section":       risk_hi,
            "mda_section":        mda_hi,
            "business_section":   biz_hi,
            "financials_section": fin_hi,
        },
        "expected_label": "high_risk",
    },
    {
        "name": "STRUCTURED | Low risk filing",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": f"{risk_lo} {mda_lo} {biz_lo} {fin_lo}",
            "risk_section":       risk_lo,
            "mda_section":        mda_lo,
            "business_section":   biz_lo,
            "financials_section": fin_lo,
        },
        "expected_label": "low_risk",
    },
    {
        "name": "STRUCTURED | Conflicting signals (high risk + low risk mixed)",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": f"{risk_hi} {mda_lo} {biz_lo} {fin_hi}",
            "risk_section":       risk_hi,   # high risk
            "mda_section":        mda_lo,    # low risk
            "business_section":   biz_lo,    # low risk
            "financials_section": fin_hi,    # high risk
        },
        # No strict expected — interesting to observe what model decides
    },
    {
        "name": "STRUCTURED | Empty sections (only text provided)",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": f"{risk_hi} {mda_hi} {biz_hi} {fin_hi}",
            "risk_section":       "",
            "mda_section":        "",
            "business_section":   "",
            "financials_section": "",
        },
        # Falls back to unstructured mode since sections are empty
    },
]


# ---------------------------------------------------------------------------
# 6. INDUSTRY-SPECIFIC REAL WORLD TESTS
# ---------------------------------------------------------------------------

industry_tests = [
    {
        "name": "INDUSTRY | Banking — credit risk + regulatory",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "Our loan portfolio faces elevated credit risk due to rising default rates. "
                "Regulatory capital requirements have increased significantly this year. "
                "Non-performing assets have grown and impairment charges are expected to rise. "
                "Liquidity risk has increased due to deposit outflows and tighter credit markets. "
                "We face investigation from regulators regarding compliance and anti-money laundering."
            )
        },
        "expected_label": "high_risk",
    },
    {
        "name": "INDUSTRY | Retail — inflation pressure + stable sales",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "Retail operations remain stable despite inflationary pressure on consumer spending. "
                "Same-store sales growth of 3% reflects resilient demand across our store network. "
                "Supply chain disruption has moderated but continues to impact inventory availability. "
                "No material litigation and positive cash flow support our near-term outlook. "
                "We are investing in digital transformation to drive long-term profitable growth."
            )
        },
    },
    {
        "name": "INDUSTRY | Energy — commodity volatility + strong output",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "Oil and gas production reached record levels with strong revenue growth this year. "
                "Commodity price volatility remains a key risk to our financial projections. "
                "Environmental regulatory compliance costs have increased but remain manageable. "
                "No material litigation and robust cash flow support continued capital investment. "
                "Exploration success and reserve additions underpin our long-term profitable outlook."
            )
        },
    },
    {
        "name": "INDUSTRY | Pharma — pipeline failure + high R&D spend",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "Two late-stage clinical trials failed to meet primary endpoints this year. "
                "R&D expenditure of $2.1B has not yet yielded approved products and cash burn is high. "
                "Patent expiry on our lead drug creates significant revenue concentration risk. "
                "Regulatory uncertainty and FDA review timelines add to uncertainty in our pipeline. "
                "Litigation from generic manufacturers threatens our intellectual property protections."
            )
        },
        "expected_label": "high_risk",
    },
    {
        "name": "INDUSTRY | SaaS — strong ARR growth + no profitability",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "Annual recurring revenue grew 45% year over year driven by strong enterprise adoption. "
                "Net revenue retention of 120% reflects robust customer expansion and low churn. "
                "The company is not yet profitable but maintains strong liquidity with two years of runway. "
                "No material litigation and strong customer satisfaction underpin our growth trajectory. "
                "We raised $300M in a Series D to accelerate market expansion and product development."
            )
        },
    },
    {
        "name": "INDUSTRY | Manufacturing — supply chain + stable demand",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "Supply chain disruption and raw material cost inflation impacted margins this year. "
                "Despite headwinds, production volumes remained stable and customer demand was robust. "
                "We have no material litigation and maintain adequate liquidity for operations. "
                "Capital investment in automation is expected to improve efficiency and reduce costs. "
                "Long-term contracts with key customers provide revenue visibility and reduce uncertainty."
            )
        },
    },
]


# ---------------------------------------------------------------------------
# 7. LINGUISTIC EDGE CASES
# ---------------------------------------------------------------------------

linguistic_tests = [
    {
        "name": "LINGUISTIC | All negations of risk words",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "We have no bankruptcy risk and no fraud concerns whatsoever in our operations. "
                "There is no litigation, no material weakness, and no going concern issues identified. "
                "We face no covenant breach, no regulatory sanctions, and no cybersecurity incidents. "
                "Not a single adverse finding was noted in our internal or external audit reviews. "
                "Neither impairment nor restructuring charges are expected in the foreseeable future."
            )
        },
        "expected_label": "low_risk",
    },
    {
        "name": "LINGUISTIC | Boilerplate disclaimer language",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "This annual report on Form 10-K contains forward-looking statements within the meaning "
                "of the Securities Exchange Act. These statements involve known and unknown risks, "
                "uncertainties and other factors which may cause actual results to differ materially. "
                "The company undertakes no obligation to update any forward-looking statements made herein. "
                "Past performance is not necessarily indicative of future results or financial condition."
            )
        },
        # Boilerplate — model may vary; no strict expectation
    },
    {
        "name": "LINGUISTIC | Heavily hedged language",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "While we may face potential challenges, management believes risks are manageable. "
                "Although uncertainty exists in certain markets, our diversified portfolio provides stability. "
                "Despite some competitive pressure, we remain confident in our ability to generate returns. "
                "Even though inflation persists, our pricing strategy has largely offset cost increases. "
                "We believe our liquidity position is adequate to meet obligations for the foreseeable future."
            )
        },
    },
    {
        "name": "LINGUISTIC | Non-English words mixed in",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "The company faces significant risque and faillite concerns in European markets. "
                "Ongoing litigation und regulatory Untersuchung pose adverse financial exposure. "
                "Material weakness in controles internos and negative cash flow threaten viability. "
                "Going concern doubts expressed by auditors create severe uncertainty for investors. "
                "Fraud allegations and covenant breach further complicate the financial outlook significantly."
            )
        },
        "expected_label": "high_risk",
    },
    {
        "name": "LINGUISTIC | Very long text (stress test)",
        "method": "POST",
        "url": f"{BASE}/predict",
        "body": {
            "text": (
                "The company reported strong revenue growth and improved profitability this fiscal year. "
                "No material litigation, no going concern issues, and robust positive cash flow. "
                "Market leadership and innovative product pipeline support long-term stable growth. "
            ) * 30  # Repeat 30x to create a long document
        },
        "expected_label": "low_risk",
    },
]


# ---------------------------------------------------------------------------
# RUN ALL TEST GROUPS
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    all_groups = [
        ("1. BASIC / INFRASTRUCTURE",       basic_tests),
        ("2. UNSTRUCTURED — HIGH RISK",      high_risk_tests),
        ("3. UNSTRUCTURED — LOW RISK",       low_risk_tests),
        ("4. UNSTRUCTURED — MEDIUM RISK",    medium_risk_tests),
        ("5. STRUCTURED MODE",               structured_tests),
        ("6. INDUSTRY-SPECIFIC",             industry_tests),
        ("7. LINGUISTIC EDGE CASES",         linguistic_tests),
    ]

    total_passed = 0
    total_failed = 0

    for group_name, tests in all_groups:
        print(f"\n{'='*70}")
        print(f"  GROUP: {group_name}")
        print(f"{'='*70}")
        run_tests(tests)