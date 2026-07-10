import io
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table


class ReportAgent:
    def __init__(self, summary_text: str, headline_metrics: dict, recommendations: list):
        self.summary_text = summary_text
        self.headline_metrics = headline_metrics
        self.recommendations = recommendations

    def create_pdf(self, output_path: str):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph('Autonomous Sales Intelligence Agent Report', styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(self.summary_text, styles['BodyText']))
        elements.append(Spacer(1, 12))

        if self.headline_metrics:
            table_data = [['Metric', 'Value']] + [[k, str(v)] for k, v in self.headline_metrics.items()]
            elements.append(Table(table_data))
            elements.append(Spacer(1, 12))

        if self.recommendations:
            elements.append(Paragraph('Recommendations', styles['Heading2']))
            for rec in self.recommendations:
                elements.append(Paragraph(f'- {rec}', styles['BodyText']))
                elements.append(Spacer(1, 6))

        doc.build(elements)
        with open(output_path, 'wb') as f:
            f.write(buffer.getvalue())
        return output_path
