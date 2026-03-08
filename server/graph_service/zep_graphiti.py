import asyncio
import logging
from typing import Annotated

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient  # type: ignore
from graphiti_core.driver.driver import GraphDriver  # type: ignore
from graphiti_core.driver.neo4j_driver import Neo4jDriver  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient, LLMConfig, OpenAIClient  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


class ZepGraphiti(Graphiti):
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        embedder: OpenAIEmbedder | None = None,
        cross_encoder: OpenAIRerankerClient | None = None,
        graph_driver: GraphDriver | None = None,
    ):
        super().__init__(
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            graph_driver=graph_driver,
        )

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


def create_graphiti_client(settings: ZepEnvDep) -> ZepGraphiti:
    # LLM configuration
    llm_api_key = settings.api_key or settings.openai_api_key
    llm_base_url = settings.base_url or settings.openai_base_url
    llm_model = settings.model_name

    logger.info(f'Initializing LLM: model={llm_model}, base_url={llm_base_url}')

    llm_config = LLMConfig(api_key=llm_api_key, base_url=llm_base_url, model=llm_model)
    llm_client = OpenAIClient(config=llm_config)

    # Embedder configuration
    emb_api_key = settings.embedding_api_key or settings.api_key or settings.openai_api_key
    emb_base_url = settings.embedding_base_url or settings.base_url or settings.openai_base_url
    emb_model = settings.embedding_model_name or 'text-embedding-3-small'

    logger.info(
        f'Initializing Embedder: model={emb_model}, dim={settings.embedding_dim}, base_url={emb_base_url}'
    )

    embedder_config = OpenAIEmbedderConfig(
        api_key=emb_api_key,
        base_url=emb_base_url,
        embedding_model=emb_model,
        embedding_dim=settings.embedding_dim,
    )
    embedder = OpenAIEmbedder(config=embedder_config)

    # Reranker configuration
    reranker_model = llm_model
    logger.info(f'Initializing Reranker: model={reranker_model}, base_url={llm_base_url}')

    reranker_config = LLMConfig(api_key=llm_api_key, base_url=llm_base_url, model=reranker_model)
    cross_encoder = OpenAIRerankerClient(config=reranker_config)

    # Database driver selection
    logger.info(f'Initializing Database: backend={settings.db_backend}')
    if settings.db_backend == 'falkordb':
        try:
            from graphiti_core.driver.falkordb_driver import FalkorDriver

            driver = FalkorDriver(
                host=settings.falkordb_host,
                port=settings.falkordb_port,
                username=settings.falkordb_user,
                password=settings.falkordb_password,
            )
        except ImportError:
            logger.error(
                "falkordb package not found. Please install with 'pip install graphiti-core[falkordb]'"
            )
            raise HTTPException(
                status_code=500, detail='FalkorDB driver dependencies missing'
            ) from None
    else:
        # Default to Neo4j
        driver = Neo4jDriver(
            uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password
        )

    return ZepGraphiti(
        llm_client=llm_client, embedder=embedder, cross_encoder=cross_encoder, graph_driver=driver
    )


async def get_graphiti(settings: ZepEnvDep):
    client = create_graphiti_client(settings)
    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    # Give the DB a moment to stabilize in container environments
    if settings.db_backend == 'falkordb':
        await asyncio.sleep(2)

    # We no longer manually call build_indices_and_constraints here
    # as the Graphiti drivers handle this automatically on initialization.
    # This prevents redundant "Index already exists" error noise in the logs.
    logger.info('Database client initialized. Indices are being managed by the driver.')
    create_graphiti_client(settings)


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
