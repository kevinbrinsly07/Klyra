
from unittest.mock import MagicMock, patch

import pytest

from app.services.document_service import DocumentService


@pytest.fixture
def service():
    return DocumentService()


@patch("app.services.document_service.PdfReader")
def test_extract_text_from_single_page(mock_reader, service):
    page = MagicMock()
    page.extract_text.return_value = "This is a test PDF."
    mock_reader.return_value.pages = [page]

    result = service.extract_text("sample.pdf")

    assert result == "This is a test PDF."


@patch("app.services.document_service.PdfReader")
def test_extract_text_from_multiple_pages(mock_reader, service):
    page1 = MagicMock()
    page1.extract_text.return_value = "First page."

    page2 = MagicMock()
    page2.extract_text.return_value = "Second page."

    mock_reader.return_value.pages = [page1, page2]

    result = service.extract_text("sample.pdf")

    assert result == "First page.\n\nSecond page."


@patch("app.services.document_service.PdfReader")
def test_empty_pages_are_skipped(mock_reader, service):
    page1 = MagicMock()
    page1.extract_text.return_value = "First page."

    page2 = MagicMock()
    page2.extract_text.return_value = ""

    page3 = MagicMock()
    page3.extract_text.return_value = "Third page."

    mock_reader.return_value.pages = [page1, page2, page3]

    result = service.extract_text("sample.pdf")

    assert result == "First page.\n\nThird page."


@patch("app.services.document_service.PdfReader")
def test_all_empty_pages_return_empty_string(mock_reader, service):
    page1 = MagicMock()
    page1.extract_text.return_value = ""

    page2 = MagicMock()
    page2.extract_text.return_value = None

    mock_reader.return_value.pages = [page1, page2]

    result = service.extract_text("sample.pdf")

    assert result == ""


@patch("app.services.document_service.PdfReader")
def test_invalid_pdf_error_is_propagated(mock_reader, service):
    mock_reader.side_effect = Exception("Invalid PDF")

    with pytest.raises(Exception, match="Invalid PDF"):
        service.extract_text("invalid.pdf")
