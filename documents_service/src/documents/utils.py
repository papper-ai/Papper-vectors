import uuid

from fastapi import UploadFile

from src.documents.schemas import CreateVaultRequest
from src.repositories.models import Vault, Document
from src.repositories.postgres_repository import DocumentRepository, VaultRepository
from src.utils.processors import read_document


async def add_vault(
    create_vault_request: CreateVaultRequest, vault_repository: VaultRepository
) -> Vault:
    id = uuid.uuid4()  # Generate random unique identifier

    vault = Vault(
        id=id,
        name=create_vault_request.vault_name,
        type=create_vault_request.vault_type,
        user_id=create_vault_request.user_id,
    )

    await vault_repository.add(vault)

    return id


async def add_document(
    file: UploadFile, vault_id: uuid, document_repository: DocumentRepository
) -> uuid:
    id = uuid.uuid4()  # Generate random unique identifier

    text = await read_document(file)

    document = Document(
        id=id,
        name=file.filename,
        text=text,
        vault_id=vault_id,
    )

    await document_repository.add(document)

    return id
