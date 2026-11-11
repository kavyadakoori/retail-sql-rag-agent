import pytest

from src.tools.sql_safety import is_destructive_request, validate_sql


def test_validate_sql_accepts_select():
    assert validate_sql("SELECT * FROM stores;") == "SELECT * FROM stores"


@pytest.mark.parametrize("statement", [
    "DELETE FROM stores",
    "SELECT * FROM stores; DROP TABLE stores",
    "UPDATE stores SET city = 'Delhi'",
])
def test_validate_sql_rejects_unsafe_statements(statement):
    with pytest.raises(ValueError):
        validate_sql(statement)


@pytest.mark.parametrize(
    "instruction",
    [
        "Delete all sales records.",
        "Please remove customer records",
        "Can you update all orders?",
        "I need to drop this table",
    ],
)
def test_destructive_natural_language_requests_are_detected(instruction):
    assert is_destructive_request(instruction)


@pytest.mark.parametrize(
    "question",
    [
        "How many records were deleted last month?",
        "What is the return policy?",
        "Show sales revenue by region.",
    ],
)
def test_read_only_questions_are_not_misclassified_as_destructive(question):
    assert not is_destructive_request(question)
