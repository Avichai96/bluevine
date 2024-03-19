import dateutil.parser as dparser
from collections import Counter
import pandas as pd
import requests
import random
import json
import logging
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_fmt = "[%(funcName)s] -> %(message)s"
file_fmt = "%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d -> %(message)s"
datefmt = "[%Y/%m/%d - %H:%M:%S]"

custom_theme = Theme({
    "logging.level.debug": "deep_sky_blue2",
    "logging.level.info": "green",
    "logging.level.warning": "yellow",
    "logging.level.error": "bold red",
    "logging.level.critical": "bold red on white"
})

class CustomRichHandler(RichHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(console=Console(theme=custom_theme), rich_tracebacks=True)

    # def emit(self, record):
    #     log_context = get_logging_context()
    #     for attr in EXTRA_ATTRIBUTES:
    #         setattr(record, attr, log_context.get(attr, '...'))
    #     super().emit(record)


'''Console setting'''
console_handler = CustomRichHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter(console_fmt, datefmt))
logger.addHandler(console_handler)

'''File setting'''
file_handler = RotatingFileHandler('answers.txt', maxBytes=10*1024*1024, backupCount=5)  # 10 MB per file, keep 5 files
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(file_fmt, datefmt))
logger.addHandler(file_handler)


def fetch_isbns(query, limit=10, output_file="books-isbns.txt"):
    """
    Fetches ISBNs from the Open Library API based on a given query.

    Parameters:
    query (str): The search query to be used for fetching ISBNs.
    limit (int, optional): The maximum number of results to fetch. Defaults to 100.
    output_file (str, optional): The name of the file to write the fetched ISBNs. Defaults to "isbns.txt".

    Returns:
    list: A list of ISBNs fetched from the Open Library API.

    Raises:
    None

    Example:
    fetch_isbns("action", limit=50, output_file="isbns.txt")
    """
    base_url = "https://openlibrary.org/search.json"
    isbns = []

    params = {"q": query, "limit": limit}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        for doc in data["docs"]:
            isbn_list = doc.get("isbn", [])
            isbns.extend(isbn_list)

        # Write the ISBNs to a file
        with open(output_file, "w") as file:
            for isbn in isbns:
                file.write(isbn + "\n")
    else:
        logger.warning(f"Failed to fetch data: {response.status_code}")

    return isbns


def fetch_book_details(isbn):
    """
    Fetches book details from the Open Library API based on the provided ISBN.

    Parameters:
    isbn (str): The ISBN of the book.

    Returns:
    dict or None: A dictionary containing the book details if the request is successful, or None if the request fails.

    """
    base_url = f"https://openlibrary.org/isbn/{isbn}.json"
    response = requests.get(base_url)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def search_item(
    query,
    limit=100,
    fields="key,title,id_goodreads,author_name,isbn,number_of_pages_median,publish_date,publish_year,publisher,first_sentence,last_modified_i",
) -> json:
    """
    Searches for items using the Open Library API.

    Parameters:
        query (str): The search query.
        limit (int, optional): The maximum number of items to retrieve. Defaults to 100.
        fields (str, optional): The fields to include in the response. Defaults to "key,title,id_goodreads,author_name,isbn,number_of_pages_median,publish_date,publish_year,publisher".

    Returns:
        dict: The JSON response containing the search results.

    """
    base_url = "https://openlibrary.org/search.json"
    params = {"q": query, "limit": limit, "fields": fields}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()


def get_books_info(isbn_file, sample_size=5):
    """
    Get books information based on a random list of 5 ISBNs from a file.

    Parameters:
        isbn_file (str): The path to the file containing the list of ISBNs.
        sample_size (int): The number of random ISBNs to select from the file.

    Returns:
        list: A list of dictionaries, where each dictionary contains information about a book.

    """
    books_data = []

    with open(isbn_file, "r") as file:
        isbns = file.readlines()

    # Select a random sample of 5 ISBNs
    random_isbns = random.sample(isbns, min(sample_size, len(isbns)))

    for isbn in random_isbns:
        isbn = isbn.strip()
        book_data = search_item(isbn)
        if book_data:
            books_data.extend(book_data["docs"])

    return books_data


# Question n. 1
def count_unique_titles(data):
    """
    Count the number of unique titles in the dataset.

    Parameters:
    data (list): A list of dictionaries, where each dictionary contains information about a book.

    Returns:
    int: The number of unique book titles in the data.
    """
    df = pd.DataFrame(data)
    return df["title"].nunique()


# Question n. 2
def find_book_with_most_isbns(data):
    """
    Find the book with the most number of different ISBNs in the dataset.

    Parameters:
    data (list): A list of dictionaries, where each dictionary contains information about a book.

    Returns:
    dict: A dictionary with the title of the book having the most ISBNs and the count of those ISBNs.
    """
    df = pd.DataFrame(data)

    # Add a column for the number of ISBNs per book
    df["isbn_count"] = df["isbn"].apply(len)

    # Find the book with the maximum number of ISBNs
    max_isbn_row = df.loc[df["isbn_count"].idxmax()]

    return {"title": max_isbn_row["title"], "isbn_count": max_isbn_row["isbn_count"]}


# Question n. 3
def count_books_without_goodreads(data):
    """
    Count the number of books in the dataset that don't have a Goodreads ID.

    Parameters:
    data (list): A list of dictionaries, each containing information about a book.

    Returns:
    int: The number of books without a Goodreads ID.
    """
    df = pd.DataFrame(data)

    # Count books where the 'id_goodreads' list is empty or not present
    count = (
        df["id_goodreads"]
        .apply(lambda ids: len(ids) == 0 if isinstance(ids, list) else True)
        .sum()
    )

    return count


# Question n. 4
def count_books_multiple_authors(data):
    """
    Count the number of books in the dataset that have more than one author.

    Parameters:
    data (list): A list of dictionaries, each containing information about a book.

    Returns:
    int: The number of books with more than one author.
    """
    df = pd.DataFrame(data)

    # Count books with more than one author
    count = (
        df["author_name"]
        .apply(lambda authors: len(authors) > 1 if isinstance(authors, list) else False)
        .sum()
    )

    return count


# Question n. 5
def count_books_per_publisher(data):
    """
    Count the number of books published per publisher.

    Parameters:
    data (list): A list of dictionaries, each containing information about a book.

    Returns:
    dict: A dictionary with publishers as the index and the number of books published as the values.
    """
    df = pd.DataFrame(data)

    # If there are multiple publishers per book, we need to explode the 'publisher' column
    if "publisher" in df.columns:
        df = df.explode("publisher")

    # Count the number of occurrences of each publisher
    books_per_publisher = df["publisher"].value_counts()

    return books_per_publisher.to_dict()


# Question n. 6
def median_number_of_pages(data):
    """
    Calculate the median number of pages for books in the given list.

    Parameters:
    data (list): A list of dictionaries, each containing information about a book, including its number of pages.

    Returns:
    float: The median number of pages among the books.
    """
    page_numbers = [
        book["number_of_pages_median"]
        for book in data
        if "number_of_pages_median" in book
    ]

    # Calculate and return the median of these page numbers
    return pd.Series(page_numbers).median()


# Question n. 7
def most_common_publish_month(data):
    """
    Find the month with the most number of published books in the given dataset.

    Parameters:
    data (list): A list of dictionaries, each containing information about a book.

    Returns:
    tuple: The month with the most publications and the number of publications in that month.
    """
    publish_months = []
    for book in data:
        for date in book.get("publish_date", []):
            try:
                parsed_date = dparser.parse(date, fuzzy=True)
                publish_months.append(parsed_date.strftime("%B"))
            except ValueError:
                pass  # Skip dates that can't be parsed

    month_counter = Counter(publish_months)
    if month_counter:
        most_common_month, most_common_count = month_counter.most_common(1)[0]
        return {
            "most common month": most_common_month,
            "most common count": most_common_count,
        }
    else:
        return {"most common month": None, "most common count": 0}


# Question n. 8
def find_longest_words_in_books(books_data):
    """
    Find the longest word in the first sentence of each book in the given books_data.

    Parameters:
    - books_data (list of dictionaries): A list of dictionaries containing information about books.

    Returns:
    - dict: A dictionary containing the longest word and the title of the book with the longest word.

    Example:
        books_data = [
            {
                "title": "Book 1",
                "first_sentence": "This is the first sentence of Book 1."
            },
            {
                "title": "Book 2",
                "first_sentence": "This is the first sentence of Book 2 with a longer word."
            },
            {
                "title": "Book 3",
                "first_sentence": "This is the first sentence of Book 3 with the longest word."
            }
        ]

        result = find_longest_words_in_books(books_data)
        print(result)
        # Output: {'longest word': 'longest', 'title with longest word': 'Book 3'}
    """
    books_df = pd.DataFrame(books_data)

    # Ensure that first_sentence is a string and handle missing values
    books_df["first_sentence"] = books_df["first_sentence"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else x
    )
    books_df["first_sentence"].fillna("", inplace=True)

    # Find the longest word in the first sentence of each book
    books_df["longest_word"] = books_df["first_sentence"].apply(
        lambda x: max(x.split(), key=len) if x.split() else ""
    )

    # Handle case where all entries might be empty
    if books_df["longest_word"].str.len().max() == 0:
        return "", ""

    # Sort the DataFrame by the length of the longest_word and get the top entry
    longest_word_df = books_df.loc[books_df["longest_word"].str.len().idxmax()]
    longest_word = longest_word_df["longest_word"]
    title_with_longest_word = longest_word_df["title"]

    return {
        "longest word": longest_word,
        "title with longest word": title_with_longest_word,
    }


# Question n. 9
def last_published_book(data):
    """
    Find the last published book in the given dataset.

    Parameters:
    data (list): A list of dictionaries, each containing information about a book.

    Returns:
    dict: Information about the last published book.
    """
    last_book = None
    latest_date = None

    for book in data:
        for date in book.get("publish_date", []):
            try:
                # Parse the date and compare with the current latest date
                parsed_date = dparser.parse(date, fuzzy=True)
                if latest_date is None or parsed_date > latest_date:
                    latest_date = parsed_date
                    last_book = book
            except ValueError:
                continue

    return last_book.get("title") if last_book else last_book


# Question n. 10
def find_most_updated_year(books_data):
    """
    Find the most updated year in the given books data.

    Parameters:
    - books_data (list of dictionaries): A list of dictionaries containing information about books.

    Returns:
    - most_updated_year (int): The most updated year in the books data.
    """
    books_df = pd.DataFrame(books_data)
    books_df["last_modified_year"] = pd.to_datetime(
        books_df["last_modified_i"], unit="s"
    ).dt.year
    most_updated_year = books_df["last_modified_year"].max()

    return most_updated_year


# Question n. 11
def get_second_published_book(data):
    """
    Determine the title of the second published book for the author with the most titles in the list.

    Parameters:
    data (list): A list of dictionaries, each containing information about a book.

    Returns:
    str: The title of the second published book for the top author, or None if not applicable.
    """
    df = pd.DataFrame(data)

    # Assuming 'publish_date' is a list of dates and 'author_name' is a list of authors for each book
    df["publish_date"] = df["publish_date"].apply(
        lambda dates: [pd.to_datetime(date, errors="coerce") for date in dates]
    )
    df = df.explode("publish_date").sort_values("publish_date")
    top_author = df.explode("author_name")["author_name"].value_counts().idxmax()

    top_author_books = df[
        df["author_name"].apply(lambda authors: top_author in authors)
    ]

    # Drop duplicates and sort by publish date to get the second published book
    unique_books = top_author_books.drop_duplicates(
        subset=["title", "publish_date"]
    ).sort_values("publish_date")

    # Check if there is at least a second book
    if len(unique_books) > 1:
        second_book_title = unique_books.iloc[1]["title"]
    else:
        second_book_title = None

    return second_book_title


# Question n. 12
def find_top_publisher_author_pair(data):
    """
    Find the pair of (publisher, author) with the highest number of books published.

    Parameters:
    data (list): A list of dictionaries, each containing information about a book.

    Returns:
    tuple: The (publisher, author) pair with the highest number of books published and the count.
    """
    df = pd.DataFrame(data)

    # If there are multiple publishers or authors, we need to explode those columns to get individual pairs
    df = df.explode("publisher")
    df = df.explode("author_name")

    # Group by publisher and author and count the number of books for each pair
    count_series = df.groupby(["publisher", "author_name"]).size()

    # Find the pair with the highest count
    top_pair = count_series.idxmax()
    top_count = count_series.max()

    return {"top pair": top_pair, "top count": top_count}


def main():
    """
    Runs various functions on a dataset of books.

    Parameters:
    None

    Returns:
    None

    The 'main' function performs the following tasks on a dataset of books:
    1. Counts the number of unique titles in the dataset.
    2. Finds the book with the most number of different ISBNs in the dataset.
    3. Counts the number of books in the dataset that don't have a Goodreads ID.
    4. Counts the number of books in the dataset that have more than one author.
    5. Counts the number of books published per publisher.
    6. Calculates the median number of pages for books in the dataset.
    7. Finds the month with the most number of published books in the dataset.
    8. Finds the longest word in the first sentence of each book in the dataset.
    9. Finds the last published book in the dataset.
    10. Finds the most updated year in the dataset.
    11. Determines the title of the second published book for the author with the most titles in the dataset.
    12. Finds the pair of (publisher, author) with the highest number of books published.

    Note: The 'main' function assumes that the dataset of books is stored in a JSON file named 'books_data.json' in the current directory.
    """

    with open("books_data.json", "r") as infile:
        books_data = json.load(infile)

    # csv_data = df.to_csv('output.csv', index=False)  # Set index=False to exclude row indices
        
    logger.info(f"1. Counts the number of unique titles in the dataset is: {count_unique_titles(books_data)}")
    logger.info(f"2. Finds the book with the most number of different ISBNs {find_book_with_most_isbns(books_data)}")
    logger.info(f"3. Counts the number of books in the dataset that don't have a Goodreads ID. {count_books_without_goodreads(books_data)}")
    logger.info(f"4. Counts the number of books in the dataset that have more than one author. {count_books_multiple_authors(books_data)}")
    logger.info(f"5. Counts the number of books published per publisher. {count_books_per_publisher(books_data)}")
    logger.info(f"6. Calculates the median number of pages for books in the dataset. {median_number_of_pages(books_data)}")
    logger.info(f"7. Finds the month with the most number of published books is: {most_common_publish_month(books_data)}")
    logger.info(f"8. Finds the longest word in the first sentence of each book: {find_longest_words_in_books(books_data)}")
    logger.info(f"9. Finds the last published book in the dataset. {last_published_book(books_data)}")
    logger.info(f"10. Finds the most updated year in the dataset. {find_most_updated_year(books_data)}")
    logger.info(f"11. Determines the title of the second published book for the author with the most titles: {get_second_published_book(books_data)}")
    logger.info(f"12. Finds the pair of (publisher, author) with the highest number of books published. {find_top_publisher_author_pair(books_data)}")


if __name__ == "__main__":
    # use this to get list of ISBNs to work with
    # query = "action"  # or any other genre/keyword
    # isbns = fetch_isbns(query)  # Fetching ISBNs and writing them to 'books-isbns.txt'

    # isbn_file = "books-isbns.txt"
    # books_data = get_books_info(isbn_file)
    # # Optionally, save this detailed data to a JSON file for later use
    # with open("books_data.json", "w") as outfile:
    #     json.dump(books_data, outfile, indent=4)

    main()
