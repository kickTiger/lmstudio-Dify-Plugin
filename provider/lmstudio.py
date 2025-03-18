import logging
from collections.abc import Mapping
import requests
from urllib.parse import urljoin

from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class LmstudioModelProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials
        if validate failed, raise exception

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        try:
            base_url = credentials.get("base_url")
            if not base_url:
                raise CredentialsValidateFailedError("Base URL is required")

            # Ensure base_url ends with /
            if not base_url.endswith("/"):
                base_url += "/"

            # Try to connect to LM Studio server
            try:
                response = requests.get(
                    urljoin(base_url, "v1/models"),
                    timeout=10  # Increased timeout for remote URLs
                )
                
                if response.status_code != 200:
                    raise CredentialsValidateFailedError(
                        f"Failed to connect to LM Studio server: {response.status_code}"
                    )
            except requests.exceptions.Timeout:
                raise CredentialsValidateFailedError(
                    "Connection timed out. If you are using a remote URL, ensure it is accessible and correctly configured."
                )
            except requests.exceptions.ConnectionError:
                raise CredentialsValidateFailedError(
                    "Failed to establish connection. Please check if the URL is correct and the server is running."
                )
            except requests.exceptions.RequestException as ex:
                raise CredentialsValidateFailedError(f"Connection error: {str(ex)}")
                
        except CredentialsValidateFailedError as ex:
            raise ex
        except Exception as ex:
            logger.exception(
                f"{self.get_provider_schema().provider} credentials validate failed"
            )
            raise CredentialsValidateFailedError(f"Failed to validate credentials: {str(ex)}")
