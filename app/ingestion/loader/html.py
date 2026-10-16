from bs4 import BeautifulSoup
import logfire

def parse_html(file_name:str):
    """
    Parses HTML content and returns a BeautifulSoup object.

    Args:
        file_name (str): The name of the HTML file to parse.
    """
    
    with logfire.span("HTML Parsing",filename=file_name):
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                html_content = file.read()
                content = BeautifulSoup(html_content, 'html.parser')

            soup=BeautifulSoup(content, 'html.parser')

            # Remove junk
            for script in soup.find_all(["script", "style", "meta", "nonscript"]):
                script.decompose()  # Remove script and style elements

            # Extract text
            text=soup.get_text(separator="\n")


            # clean whitespace
            lines=(line.strip() for line in text.splitlines())
            chunks=(phrase.strip() for line in lines for phrase in line.split("  "))
            text_clean="\n".join(chunk for chunk in chunks if chunk)


            return text_clean

        except Exception as e:
            logfire.error(f"Error parsing HTML file {file_name}: {e}")
            raise e