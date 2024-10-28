*Description*
Build a dashboard that allows users to visualize datasets interactively, filtering and exploring data through various visualizations.

*Key Features*
Use libraries like Pandas for data manipulation and Plotly or Bokeh for visualization.
Implement user controls for filtering (e.g., dropdowns, sliders).
Allow users to upload datasets and visualize them dynamically.

*Learning Outcomes*
Learn how to integrate data visualization libraries with Flask.
Practice handling user input and rendering dynamic content.

------

- Create a web application to upload datasets (CSV files) and visualize them interactively.
- Set up a virtual environment to isolate dependencies, ensuring package compatibility.
- Install Flask for web development, Pandas for data manipulation, and Plotly for visualization.
- Define the main application structure in `app.py`, setting up a Flask route for rendering the web interface.
- Create an HTML template (`index.html`) to handle file uploads and display visualizations.
- Use a form in the HTML template for CSV uploads with proper encoding for file handling.
- Modify the Flask route to accept POST requests for file submissions.
- Read uploaded CSV files into Pandas DataFrames for easy data manipulation.
- Generate visualizations, like scatter plots, using Plotly based on numeric columns in the DataFrame.
- Convert Plotly visualizations into HTML and pass them to the template for rendering.
- Add optional user controls (dropdowns, sliders) in the template to filter data interactively.
- Run the application on a local server and test by uploading sample CSV files for visualization.
- Visualizations are displayed in real-time, enhancing user interaction with datasets.
- Users gain experience in handling file uploads, rendering dynamic content, and integrating data visualization with Flask.
- The project emphasizes user-friendly interfaces for data exploration, applicable to real-world scenarios.
