import logfire

def parse_text(file_name: str):
    """
    Parses text content and returns a string.

    Args:
        file_name (str): The name of the text file to parse.
    """
    
    with logfire.span("Text Parsing", filename=file_name):
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                text = file.read()

            if not text.strip():
                logfire.warning(f"No text found in text file {file_name}.")
            else:
                logfire.info(f"Successfully parsed text file {len(text)} characters.")


            return text

        except Exception as e:
            logfire.error(f"Error parsing text file {file_name}: {e}")
            raise e 