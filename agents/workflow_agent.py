import logging
from langgraph import Graph, Node, Edge

logger = logging.getLogger(__name__)


class WorkflowAgent:
    def __init__(self):
        self.graph = Graph(name='Sales Intelligence Workflow')
        self._build_graph()

    def _build_graph(self):
        self.upload_node = Node('Upload Dataset', description='Receive the uploaded CSV dataset.')
        self.profile_node = Node('Profiling Agent', description='Run dataset profiling and quality checks.')
        self.clean_node = Node('Cleaning Agent', description='Clean duplicates, missing values, and standardize columns.')
        self.eda_node = Node('EDA Agent', description='Generate sales, product, region, and customer analytics.')
        self.insight_node = Node('Insight Agent', description='Create AI-generated executive insights and risk summaries.')
        self.recommend_node = Node('Recommendation Agent', description='Produce strategic business recommendations.')
        self.report_node = Node('Report Agent', description='Generate PDF report and export charts.')
        self.forecast_node = Node('Forecasting Agent', description='Produce short-term and quarterly forecasts.')
        self.chat_node = Node('Chat Agent', description='Handle conversational analytics queries.')

        self.graph.add_node(self.upload_node)
        self.graph.add_node(self.profile_node)
        self.graph.add_node(self.clean_node)
        self.graph.add_node(self.eda_node)
        self.graph.add_node(self.insight_node)
        self.graph.add_node(self.recommend_node)
        self.graph.add_node(self.report_node)
        self.graph.add_node(self.forecast_node)
        self.graph.add_node(self.chat_node)

        self.graph.add_edge(Edge(self.upload_node, self.profile_node, label='upload -> profile'))
        self.graph.add_edge(Edge(self.profile_node, self.clean_node, label='profile -> clean'))
        self.graph.add_edge(Edge(self.clean_node, self.eda_node, label='clean -> eda'))
        self.graph.add_edge(Edge(self.eda_node, self.insight_node, label='eda -> insight'))
        self.graph.add_edge(Edge(self.insight_node, self.recommend_node, label='insight -> recommend'))
        self.graph.add_edge(Edge(self.recommend_node, self.report_node, label='recommend -> report'))
        self.graph.add_edge(Edge(self.clean_node, self.forecast_node, label='forecast request'))
        self.graph.add_edge(Edge(self.clean_node, self.chat_node, label='question request'))

    def get_graph(self):
        return self.graph

    def describe(self):
        nodes = [f'{node.name}: {node.description}' for node in self.graph.nodes]
        edges = [f'{edge.source.name} -> {edge.target.name}: {edge.label}' for edge in self.graph.edges]
        return {'nodes': nodes, 'edges': edges}
