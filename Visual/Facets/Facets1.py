from IPython.core.display import display, HTML
import pandas as pd

from sklearn.datasets import load_boston
boston_data = load_boston()

df = pd.DataFrame(boston_data['data'], columns=boston_data['feature_names'])
jsonstr = df.to_json(orient='records')

HTML_TEMPLATE = \
    """
        <link rel="import" href="/nbextensions/facets-dist/facets-jupyter.html">
        <facets-dive id="elem_id" height="600"></facets-dive>
        <script>
          var data = {jsonstr};
          document.querySelector("#elem_id").data = data;
        </script>
    """
html = HTML_TEMPLATE.format(jsonstr=jsonstr)
display(HTML(html))