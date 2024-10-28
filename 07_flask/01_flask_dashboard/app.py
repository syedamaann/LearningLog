from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1])  # Change to your preferred plot
            graph_html = pio.to_html(fig, full_html=False)
            return render_template('index.html', graph_html=graph_html)
    return render_template('index.html', graph_html=None)

if __name__ == '__main__':
    app.run(debug=True)

