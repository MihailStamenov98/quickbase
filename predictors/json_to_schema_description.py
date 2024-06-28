import json


def get_table_description(table):
    table_description = []
    table_name = table.get("name", "Unnamed Table")
    table_description.append(f"Table: {table_name}")

    fields = table.get("fields", [])
    for field in fields:
        field_name = field.get("name", "Unnamed Field")
        field_type = field.get("type", "Unknown Type")
        field_description = f"  - Field: {field_name} (Type: {field_type})"

        if field_type == "TC" and "choices" in field:
            choices = field["choices"]
            field_description += f" Choices: {', '.join(choices)}"

        if "formula" in field:
            formula = field["formula"]
            field_description += f" Formula: {formula}"

        if "lookup" in field:
            lookup = field["lookup"]
            foreign_key_field = lookup.get(
                "foreignKeyField", "Unknown Foreign Key Field"
            )
            parent_table = lookup.get("parentTable", "Unknown Parent Table")
            parent_field = lookup.get("parentField", "Unknown Parent Field")
            field_description += (
                f" Lookup: {foreign_key_field} -> {parent_table}.{parent_field}"
            )

        if "summary" in field:
            summary = field["summary"]
            aggregation = summary.get("aggregation", "Unknown Aggregation")
            field_description += f" Summary: {aggregation}"
            if "childForeignKeyField" in summary:
                child_foreign_key_field = summary["childForeignKeyField"]
                child_table = summary.get("childTable", "Unknown Child Table")
                field_description += f" (Child Foreign Key Field: {child_foreign_key_field}, Child Table: {child_table})"

        table_description.append(field_description)
    return "\n".join(table_description)


def get_tables_descriptions(json_data):
    table_descriptions = []
    tables = json_data.get("ai_dict", {}).get("tables", [])

    for table in tables:
        table_description = get_table_description(table)
        table_descriptions.append(table_description)
        table_descriptions.append("\n")
    return table_descriptions


def json_to_schema_description(json_data, text=True):
    schema_description = []
    app_name = json_data.get("ai_dict", {}).get("name", "Unknown App")
    description = json_data.get("ai_dict", {}).get(
        "description", "No description available."
    )
    schema_description.append(f"App Name: {app_name}")
    schema_description.append(f"Description: {description}\n")
    schema_description = "\n".join(schema_description)
    tables_descriptions = get_tables_descriptions(json_data)
    tables_descriptions.insert(0, schema_description)
    return "\n".join(tables_descriptions) if text else tables_descriptions
