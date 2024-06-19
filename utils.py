from bs4 import BeautifulSoup
import html


def clean_article(html_content):
    soup = BeautifulSoup(html_content, 'html5lib')

    # Remove all script and style elements
    for tag_to_remove in soup(["script", "style"]):
        tag_to_remove.decompose()

    text = soup.get_text()
    clean_text = html.unescape(text)
    return clean_text


if __name__ == "__main__":
    test_html = """<script>test script</script>
    <style type="text/css">
    p.p1 {margin: 0.0px 0.0px 0.0px 0.0px; font: 12.0px Arial; color: #333333} p.p2 {margin: 0.0px 0.0px 0.0px 0.0px;
    font: 9.0px Helvetica; color: #484848} span.s1 {font: 10.0px Helvetica; color: #6f73e8}
    </style>
    <p>Urban areas typically have high population densities. The variance and affect of different allocation types is
    minimal because there is a larger chance of an equal distribution of the population.
    </p><p><br></p><p><br></p><p><br></p><p><br></p><hr><p><em>
    <span style="color: rgb(209, 213, 216);">https://analytics.placer.ai/?data-elevio-article=133</span></em></p>
    """

    print(clean_article(test_html))