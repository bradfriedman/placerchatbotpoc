from bs4 import BeautifulSoup
import html

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html5lib')
    text = soup.get_text()
    clean_text = html.unescape(text)
    return clean_text