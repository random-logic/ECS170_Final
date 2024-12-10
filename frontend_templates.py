# List of genres for input

genres = [
    'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'Game-Show',
    'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV',
    'Romance', 'Sci-Fi', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western'
]

# Main Page Template

main_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genre Selection Form</title>
    <style>
        /* Reset margins and paddings */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* Set full-page layout */
        body {
            height: 100vh;
            width: 100vw;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: Arial, sans-serif;
            background-color: #f9f9f9; /* Optional: Change background color */
        }

        /* Form container */
        .form-container {
            width: 90%; /* Takes most of the page's width */
            max-width: 1200px; /* Restrict to a readable width */
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Center title */
        h1 {
            text-align: center;
            margin-bottom: 20px;
        }

        /* Section spacing */
        h2 {
            margin-top: 20px;
        }

        /* Genres layout */
        .genres {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
        }

        label {
            font-size: 14px;
            display: inline-block;
        }

        /* Button styling */
        button, input[type="submit"] {
            background-color: #007BFF;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }

        button:hover, input[type="submit"]:hover {
            background-color: #0056b3;
        }

        /* Input styling */
        input[type="number"] {
            width: 100%;
            padding: 8px;
            font-size: 14px;
            margin-top: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }

        /* Space between form sections */
        .form-section {
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <h1>FLICK FINDER</h1>

        <!-- Button for additional information -->
        <p style="text-align: center;">
            <button onclick="window.location.href='/info'">Click here for more information</button>
        </p>

        <!-- Genre Selection Form -->
        <form method="POST" action="/output">
            <div class="form-section">
                <h2>Select Genres (At least 1)</h2>
                <div class="genres">
                    {% for chunk in genre_chunks %}
                    {% for genre in chunk %}
                    <div>
                        <input type="checkbox" id="genre_{{ genre }}" name="genres" value="{{ genre }}">
                        <label for="genre_{{ genre }}">{{ genre }}</label>
                    </div>
                    {% endfor %}
                    {% endfor %}
                </div>
            </div>

            <div class="form-section">
                <h2><label for="avg_rating">Enter an Avg. Rating (0-10):</label></h2>
                <input 
                    type="number" 
                    id="avg_rating" 
                    name="avg_rating" 
                    step="0.1" 
                    min="0" 
                    max="10" 
                    required
                >
            </div>

            <div class="form-section">
                <h2><label for="votes_num">Enter Number of Votes (0-10000):</label></h2>
                <input 
                    type="number" 
                    id="votes_num" 
                    name="votes_num" 
                    min="0" 
                    max="10000" 
                    required
                >
            </div>

            <!-- Submit Button -->
            <div style="text-align: center;">
                <input type="submit" value="Submit">
            </div>
        </form>
    </div>
</body>
</html>
"""

# Output Page Template
output_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Output Page</title>
    <style>
        /* Reset margins and paddings */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* Full-page layout */
        body {
            height: 100vh;
            width: 100vw;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            text-align: center;
        }

        /* Output container styling */
        .output-container {
            width: 90%;
            max-width: 800px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Title styling */
        h1 {
            margin-bottom: 20px;
            color: #333;
        }

        /* Error message styling */
        h2 {
            margin-bottom: 20px;
        }

        .error-message {
            color: red;
        }

        /* Recommendation output */
        .recommendations {
            margin-top: 20px;
            font-size: 16px;
            color: #555;
            line-height: 1.5;
        }

        /* Button styling */
        button {
            background-color: #007BFF;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 20px;
        }

        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="output-container">
        <h1>Recommendations</h1>

        <!-- Error message -->
        {% if error_message %}
        <h2 class="error-message">{{ error_message }}</h2>
        {% else %}
        <!-- Recommendations section -->
        <h2>Your Recommendations</h2>
        <div class="recommendations">
            {{ output|safe }}
        </div>
        {% endif %}

        <!-- Back button -->
        <button onclick="window.location.href='/'">Go back to the main page</button>
    </div>
</body>
</html>
"""

# Info Page Template
info_page = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Info Page</title>
    <style>
        /* Reset margins and paddings */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* Full-page layout */
        body {
            height: 100vh;
            width: 100vw;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            text-align: center;
        }

        /* Info container styling */
        .info-container {
            width: 90%;
            max-width: 800px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Title styling */
        h1 {
            margin-bottom: 20px;
            color: #333;
        }

        /* List styling */
        ul {
            list-style-type: disc;
            margin: 20px 0;
            padding-left: 20px;
            text-align: left;
        }

        li {
            margin-bottom: 10px;
            font-size: 16px;
            color: #555;
        }

        p {
            font-size: 16px;
            color: #666;
            margin-bottom: 20px;
        }

        /* Button styling */
        button {
            background-color: #007BFF;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }

        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="info-container">
        <h1>About This Application</h1>
        <p>This application allows users to:</p>
        <ul>
            <li>Select their favorite genres from a predefined list.</li>
            <li>Input an average rating (a floating-point number between 0 and 10).</li>
            <li>Specify the number of votes (a positive integer).</li>
        </ul>
        <p>
            Once you submit the form, the selected genres, average rating, and number of votes will be used to display recommendations on a separate page.
        </p>
        <h1>
            <button onclick="window.location.href='/'">Go back to the main page</button>
        </h1>
    </div>
</body>
</html>
"""