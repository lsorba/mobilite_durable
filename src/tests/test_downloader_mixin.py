import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from src.utils.downloader_mixin import DownloaderMixin, DownloadStatus


class TestDownloaderMixin(unittest.TestCase):
    """
    Test the DownloaderMixin class with a mock HTTP response.

    Author: Laurent Sorba
    """

    @patch("requests.get")
    def test_download_file_success(self, mock_get):
        # Mock the response for a successful download
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1024"}
        mock_response.iter_content.return_value = [b"data"] * 256  # Simulate 1024 bytes of data
        mock_get.return_value = mock_response

        destination = Path("test_file.txt")
        status, filename = DownloaderMixin.download_file(
            "http://example.com/test_file.txt", destination
        )

        assert status == DownloadStatus.DOWNLOADED
        assert filename == destination.name
        assert os.path.exists(destination)

        # Clean up
        os.remove(destination)

    @patch("requests.get")
    def test_download_file_skip_existing(self, mock_get):
        # Create a dummy file to simulate an existing file
        destination = Path("existing_file.txt")
        with open(destination, "w") as f:
            f.write("This file already exists.")

        status, filename = DownloaderMixin.download_file(
            "http://example.com/existing_file.txt", destination
        )

        assert status == DownloadStatus.SKIPPED
        assert filename == destination.name

        # Clean up
        os.remove(destination)

    @patch("requests.get")
    def test_download_file_http_error(self, mock_get):
        # Mock the response for an HTTP error (e.g., 404)
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
        mock_get.return_value = mock_response

        destination = Path("error_file.txt")
        status, filename = DownloaderMixin.download_file(
            "http://example.com/error_file.txt", destination
        )

        assert status == DownloadStatus.ERROR
        assert filename == destination.name
        assert not os.path.exists(destination)  # Ensure the file was not created

    @patch("requests.get")
    def test_download_file_retry_after_429(self, mock_get):
        # First response: 429 Too Many Requests
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.raise_for_status.side_effect = requests.exceptions.HTTPError

        # Second response: 200 OK
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.headers = {"content-length": "1024"}
        mock_response_200.iter_content.return_value = [
            b"data"
        ] * 256  # Simulate 1024 bytes of data

        # Configure mock to return 429 first, then 200
        mock_get.side_effect = [mock_response_429, mock_response_200]

        destination = Path("retry_file.txt")
        status, filename = DownloaderMixin.download_file(
            "http://example.com/retry_file.txt", destination
        )

        # Verify that get was called twice
        assert mock_get.call_count == 2
        assert status == DownloadStatus.DOWNLOADED
        assert filename == destination.name
        assert os.path.exists(destination)

        # Clean up
        os.remove(destination)


if __name__ == "__main__":
    unittest.main()
