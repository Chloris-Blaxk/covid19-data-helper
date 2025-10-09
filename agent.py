import data_analyzer
import re

def parse_country(query: str, default='US'):
    """
    Parses a country name from the query using a simple regex.
    This is a placeholder for a more sophisticated NLP entity extraction.
    """
    # A simple regex to find capitalized words that might be country names
    # This is very basic and will need improvement.
    match = re.search(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', query)
    if match:
        # A real implementation would check this against a list of known countries
        return match.group(1)
    return default

def run_agent(query: str):
    """
    Parses a natural language query and calls the appropriate analysis function.
    Returns a dictionary with the result.
    """
    original_query = query
    query = query.lower()

    if 'top countries' in query and ('plot' in query or 'chart' in query):
        top_n_match = re.search(r'top (\d+)', query)
        n = int(top_n_match.group(1)) if top_n_match else 10
        image_path = data_analyzer.plot_top_countries_by_confirmed_cases(top_n=n)
        return {'type': 'image', 'content': image_path}

    elif 'daily cases' in query and ('plot' in query or 'chart' in query):
        country = parse_country(original_query)
        image_path = data_analyzer.plot_daily_cases_for_country(country)
        return {'type': 'image', 'content': image_path}

    elif 'mortality rate' in query:
        country = parse_country(original_query)
        result_text = data_analyzer.calculate_mortality_rate(country)
        return {'type': 'text', 'content': result_text}

    elif 'growth rate' in query:
        country = parse_country(original_query)
        result_text = data_analyzer.calculate_daily_growth_rate(country)
        return {'type': 'text', 'content': result_text}

    elif 'deaths' in query:
        date_match = re.search(r'(\d{1,2})[./](\d{1,2})', query)
        if date_match:
            month, day = int(date_match.group(1)), int(date_match.group(2))
            country = parse_country(original_query)
            result_text = data_analyzer.get_deaths_on_date(country, month, day)
            return {'type': 'text', 'content': result_text}
        else:
            return {'type': 'text', 'content': "Sorry, I can only fetch deaths for a specific date (e.g., 'deaths in US on 9/21')."}
        
    else:
        return {'type': 'text', 'content': "Sorry, I didn't understand that. Please ask about 'top countries', 'daily cases', 'mortality rate', or 'growth rate'."}

if __name__ == '__main__':
    # Example usage for testing:
    result = run_agent("Plot a chart of the top 10 countries")
    print(result)
    result = run_agent("What is the mortality rate in China")
    print(result)
    result = run_agent("Show me the deaths in US on 9/11")
    print(result)
