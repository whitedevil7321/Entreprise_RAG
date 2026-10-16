import logfire
from unstructured.partition.auto import partition

def parse_office(file_name: str):
    """
    Parses office content and returns a string.

    Args:
        file_name (str): The name of the office file to parse.
    """
    
    with logfire.span("Office Parsing", filename=file_name):
        try:
            elements = partition(file_name)
            text = "\n".join([element.text for element in elements if element.text])

            if not text.strip():
                logfire.warning(f"No text found in Office file {file_name}.")
            else:
                logfire.info(f"Successfully parsed Office file {len(text)} characters.")


            return text

        except Exception as e:
            logfire.error(f"Error parsing Office file {file_name}: {e}")
            raise e