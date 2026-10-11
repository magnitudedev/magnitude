from magnitude_benchmarks.fixtures.bfcl import normalize_schema


def test_optional_extension_is_represented_by_required_list():
    schema = {
        "type": "dict",
        "properties": {
            "name": {"type": "string"},
            "rating": {"type": "float"},
        },
        "required": ["name"],
        "optional": ["rating"],
    }
    assert normalize_schema(schema) == {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "rating": {"type": "number"},
        },
        "required": ["name"],
    }


def test_undeclared_required_names_are_removed():
    schema = {
        "type": "object",
        "properties": {
            "population": {
                "type": "object",
                "description": "Adults, children, and single residents.",
                "required": ["adults", "children", "singles"],
            }
        },
        "required": ["population", "missing"],
    }
    assert normalize_schema(schema) == {
        "type": "object",
        "properties": {
            "population": {
                "type": "object",
                "description": "Adults, children, and single residents.",
            }
        },
        "required": ["population"],
    }


def test_string_format_stays_in_description_without_grammar_constraint():
    schema = {
        "type": "string",
        "format": "date",
        "description": "Date in YYYY-MM-DD form",
    }
    assert normalize_schema(schema) == {
        "type": "string",
        "description": "Date in YYYY-MM-DD form",
    }
