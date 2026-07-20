*** Settings ***
Documentation     Smoke-level AI quality checks for the benefits assistant.
...               Mirrors the pytest evaluation suite at a business-readable
...               level — the layer stakeholders and manual QA review.
Library           ChatbotKeywords.py

*** Test Cases ***
PTO Accrual Question Is Answered And Grounded
    Ask Question    How many PTO days do full-time employees get per year?
    Answer Should Contain    20
    Answer Should Be Grounded    0.8
    Answer Should Have No Hallucinated Claims

Medical Plan Question Is Answered And Grounded
    Ask Question    Which medical plan includes dental and vision coverage?
    Answer Should Contain    Premium
    Answer Should Be Grounded    0.8
    Answer Should Have No Hallucinated Claims

Retirement Match Question Is Answered And Grounded
    Ask Question    What is the company 401(k) match percentage?
    Answer Should Contain    4%
    Answer Should Be Grounded    0.8
    Answer Should Have No Hallucinated Claims

Out Of Scope Question Gets The Exact Fallback
    Ask Question    What is the parental leave policy?
    Answer Should Be The Fallback
