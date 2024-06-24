import json

def json_to_schema_description(json_data):
    schema_description = []
    
    app_name = json_data.get("ai_dict", {}).get("name", "Unknown App")
    description = json_data.get("ai_dict", {}).get("description", "No description available.")
    
    schema_description.append(f"App Name: {app_name}")
    schema_description.append(f"Description: {description}\n")
    
    tables = json_data.get("ai_dict", {}).get("tables", [])
    
    for table in tables:
        table_name = table.get("name", "Unnamed Table")
        schema_description.append(f"Table: {table_name}")
        
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
                foreign_key_field = lookup.get("foreignKeyField", "Unknown Foreign Key Field")
                parent_table = lookup.get("parentTable", "Unknown Parent Table")
                parent_field = lookup.get("parentField", "Unknown Parent Field")
                field_description += f" Lookup: {foreign_key_field} -> {parent_table}.{parent_field}"
            
            if "summary" in field:
                summary = field["summary"]
                aggregation = summary.get("aggregation", "Unknown Aggregation")
                field_description += f" Summary: {aggregation}"
                if "childForeignKeyField" in summary:
                    child_foreign_key_field = summary["childForeignKeyField"]
                    child_table = summary.get("childTable", "Unknown Child Table")
                    field_description += f" (Child Foreign Key Field: {child_foreign_key_field}, Child Table: {child_table})"
            
            schema_description.append(field_description)
        
        schema_description.append("\n")
    
    return "\n".join(schema_description)

#with open('app1.json') as f:
#    json_data = json.load(f)
#
#schema_description = json_to_schema_description(json_data)
