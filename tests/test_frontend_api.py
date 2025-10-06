# tests/test_frontend.py
from bs4 import BeautifulSoup

def test_index_html_has_core_elements():
    with open("index.html", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    assert soup.find("h1").text == "Monitor de Qualidade do Ar"
    assert soup.find("button", string=lambda x: "Iniciar" in x)
    assert soup.find("img", {"id": "camera"})
    assert soup.find("div", {"id": "report"})
