import os
import pandas as pd
from tqdm import tqdm

import re
import requests
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from urllib.parse import urlparse


from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean
from unstructured.cleaners.core import clean_ordered_bullets, clean_bullets, clean_non_ascii_chars  #tbd


class WebScraper:
    def __init__(self, file, use_split):
        self.file = file
        self.use_split = use_split
     
    def scrape_select(self, url, url_hash):
        scraper = ScrapeAll(url, url_hash)
        scraper.scrape_other()
        scraper.clean_unstructured()
        scraper.concatenate_strings()
        scraper.standardize_chunks() 
        scraper.store_as_text()
        

    def scrape(self):

        df = pd.read_csv(self.file)
        urls_with_hashes = []
        scrape_dir  = '/home/ec2-user/ITMT597/chatbot/web_scrapping/scrapped_data/'
        
        if not os.path.exists(scrape_dir):
            os.mkdir(scrape_dir)
        os.chdir(scrape_dir)

        for index, row in tqdm(df.iterrows()):
            url = row['urls']
            url_hash = hash(url)
            """this condition requires creation of split_id column in the url csv
            after the creation of split_id column, the default value of split_id should be 'unspecified'
            however for urls of interest, curation is needed from the user to unset the default value"""

            # Extract data from the current URL and save it to a text file
            self.scrape_select(url, url_hash)
            urls_with_hashes.append([url, url_hash])

        #now combining all the text files into one file
        output_file = "combined_all.txt"
    
        # Create a list to store the content of each file
        file_contents = []

        # Loop through all .txt files in the directory
        for filename in os.listdir(scrape_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(scrape_dir, filename)
                
                # Open and read the content of each .txt file
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_contents.append(file.read())

        # Combine the contents of all files into one string
        combined_text = '\n'.join(file_contents)

        # Write the combined text to the output file
        output_file_path = os.path.join(dir_path, output_file)
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(combined_text)

        print(f"Combined text file for {dir_name} urls saved to {output_file}")

        #convert urls_with_hashes to dataframe
        df = pd.DataFrame(urls_with_hashes, columns=['urls', 'hashes', 'scrape_type'])
        df.to_csv(f"{scrape_dir}urls_with_hashes.csv", index=False)   

    #IIT webpages don't have headers and footers defined via respective html tags
    #rather for headers and footers specific types of divs are being used
    #in such a scenario unstructured's skip_header_footers does not work, hence this:   
    # to do
    def is_html(self, url):
    #Returns True if the response content is HTML, False otherwise.
        retry_strategy = Retry(
        total=8,  # Number of maximum retries
        backoff_factor=1,  # Exponential backoff factor
        status_forcelist=[500, 502, 503, 504],  # HTTP status codes to retry on
        )

        # Create an HTTP session with retry settings
        http = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http.mount("http://", adapter)
        http.mount("https://", adapter)

        try:
        # Send a GET request using the HTTP session
        #here 5 sec is connection_timeout where 27 sec is read time_out once the connection is established
            response = requests.get(url, timeout=(5, 27))
            content_type = response.headers['Content-Type']    
            pattern = "html"

            # Use re.search() to check if the pattern is in the string
            if re.search(pattern, content_type):
                return True
            else:
                return False
        except requests.exceptions.RequestException as e:
            print(f"Error while fetching URL: {url}. Exception: {e}")
    
    def optional_rm_headers_footers(self, url, new_list):
        if self.is_html(url):
            self.new_list = new_list[1:-1]
        else:
            pass

class ScrapeIIT:
    def __init__(self, url, url_hash):
        self.url = url
        self.url_hash = url_hash
        
    # Function for running through IIT URL's
    def scrape_iit(self):

        retry_strategy = Retry(
        total=8,  # Number of maximum retries
        backoff_factor=1,  # Exponential backoff factor
        status_forcelist=[500, 502, 503, 504],  # HTTP status codes to retry on
        )

        # Create an HTTP session with retry settings
        http = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http.mount("http://", adapter)
        http.mount("https://", adapter)

        try:
        # Send a GET request using the HTTP session
        #here 5 sec is connection_timeout where 27 sec is read time_out once the connection is established
            response = requests.get(self.url, timeout=(5, 27))

            # Empty dictionary for storing headers and tables
            data = {}

            sos_added = True

            if response.status_code == 200:

                soup = BeautifulSoup(response.text, 'html.parser')

                main = soup.find('main')

                if main:
                    first_heading = None
                    heading_text = None

                    # Find all relevant elements within <main>
                    for element in main.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table', 'ul','div']): #removed 'a'

                        if element.find_parent('nav'):
                            continue

                        tag_name = element.name

                        if tag_name.startswith('h'):
                            heading_text = element.text.strip()
                            if heading_text:
                                if first_heading is None:
                                    first_heading = f'sos: {heading_text}'
                                    data[heading_text] = []
                                else:
                                    data[heading_text] = []

                        elif tag_name == 'p':
                            if heading_text:
                                passage_text = element.text.strip()
                                if heading_text not in data:
                                    data[heading_text] = []
                                data[heading_text].append(passage_text)

                        elif tag_name == 'ul':
                            list_data = []
                            for li in element.find_all('li'):
                                bullet_point = li.text.strip()
                                list_data.append(bullet_point)

                            if heading_text:
                                data[heading_text].extend(list_data)

                        elif tag_name == 'table':
                            table_data = []
                            for row in element.find_all('tr'):
                                row_data = [cell.text.strip() for cell in row.find_all('td')]
                                table_data.append(row_data)
                            if heading_text:
                                data[heading_text].append(table_data)

                        elif tag_name == 'span' and 'profile-item__contact__item' in element.get('class', []):
                            # Extract data from the location element
                            info_type = element.find('i')
                            if info_type:
                                info_type = info_type['class'][1]
                                info_text = element.get_text(strip=True)
                                last_word = element['class'][-1]
                                data[heading_text].append(f'{last_word}: {info_text}')

                    if first_heading:
                        modified_data = {}
                        for key in data:
                            if key != first_heading:
                                modified_key = f'{first_heading} <{key}>'
                                modified_data[modified_key] = data[key]
                            else:
                                modified_data[first_heading] = data[key]

                    output_file_path = f'{self.url_hash}.txt'
                    
                    # Find all relevant elements
                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        for heading, content in modified_data.items():
                            output_file.write(f"{heading}\n")
                            for item in content:
                                if isinstance(item, str):
                                    output_file.write(f"{item}\n")
                                elif isinstance(item, list):
                                    for row in item:
                                        output_file.write(f"{', '.join(row)}\n")
                            output_file.write(f"Information Source: {self.url}\n\n")

            else:
                print(f"Failed to retrieve URL: {self.url}. Status code: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Error while fetching URL: {self.url}. Exception: {e}")

    # Function for running through Bulletin URL's
    def scrape_bulletin(self):
        retry_strategy = Retry(
        total=8,  # Number of maximum retries
        backoff_factor=1,  # Exponential backoff factor
        status_forcelist=[500, 502, 503, 504],  # HTTP status codes to retry on
        )

        # Create an HTTP session with retry settings
        http = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http.mount("http://", adapter)
        http.mount("https://", adapter)

        try:
        # Send a GET request using the HTTP session
        #here 5 sec is connection_timeout where 27 sec is read time_out once the connection is established
            response = requests.get(self.url, timeout=(5, 27))

            # Empty dictionary for storing headers and tables
            data = {}

            sos_added = True

            if response.status_code == 200:

                soup = BeautifulSoup(response.text, 'html.parser')

                heading_text = None

                # Find all relevant elements within <main>
                for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table','ul','div']):
                    tag_name = element.name

                    if tag_name == 'h1':
                        heading_text1 = element.text.strip()
                        if heading_text1:
                            main_heading = heading_text1

                    if tag_name.startswith('h'):
                        heading_text = element.text.strip()
                        previous_heading = heading_text
                        if heading_text:
                            if sos_added:
                                if tag_name == 'h1':
                                    heading_text = f'sos: {heading_text}'
                                else:
                                    if main_heading is None:
                                        main_heading = 'Academic Programs Details'
                                    heading_text = f'sos: {main_heading} <{heading_text}>'
                            data[heading_text] = []

                    elif tag_name == 'p':
                        if heading_text:
                            passage_text = element.text.strip()
                            data[heading_text].append(passage_text)

                    elif tag_name == 'ul':
                        list_data = []
                        for li in element.find_all('li'):
                            bullet_point = li.text.strip()
                            list_data.append(bullet_point)

                        if heading_text:
                            data[heading_text].extend(list_data)

                    elif tag_name == 'table':
                        table_data = []
                        a = False
                        for row in element.find_all('tr'):
                            row_data = [cell.text.strip() for cell in row.find_all(['th', 'td'])]
                            if row_data == ['Year 1']:
                                new_data = []
                                new_data.append(row_data)  # Start a new list when 'Year 1' is encountered
                                a = True
                            elif a:
                                new_data.append(row_data)  # Append subsequent rows to the new list
                            else:
                                table_data.append(row_data)
                        if a:
                            data1 = new_data
                            last_item = data1[-1]
                            columns = data1[1]
                            df1 = pd.DataFrame(data1, columns=columns)

                            year_word = None
                            for index, row in df1.iterrows():
                                if row[columns[0]].startswith('Year'):
                                    year_word = row[columns[0]]

                                elif row[columns[0]].startswith('Semester'):
                                    df1.at[index, columns[0]] = f'{year_word}\n{row[columns[0]]}'
                                    df1.at[index, columns[2]] = f'{year_word}\n{row[columns[2]]}'

                            df1 = df1.replace('None', pd.NA).dropna()
                            new_df = df1.iloc[:, -2:]
                            df1 = df1.iloc[:, :2]
                            column_names = df1.columns
                            new_df.columns = column_names
                            result_df = pd.concat([df1, new_df], axis=0)
                            result_list_of_lists = result_df.values.tolist()
                            result_list_of_lists.append(last_item)
                            for item in result_list_of_lists:
                                table_data.append(item)
                        if heading_text:
                            if any(char.isalpha() for char in 'table_data[0][0]') and any(char.isdigit() for char in 'table_data[0][0]'):
                                intro_text = f"These are the {previous_heading} courses for the {main_heading}"
                                table_data.insert(0, [intro_text.replace("sos: ", "")])
                                #table_data.pop(1)
                                data[heading_text].append(table_data)
                            else:
                                intro_text = f"These are the {table_data[0][0]} for the {main_heading} {heading_text} and the total credits are {table_data[0][1]}"
                                table_data.insert(0, [intro_text.replace("sos: ", "")])
                                #table_data.pop(2)
                                table_data.pop(1)
                                data[heading_text].append(table_data)

                    elif tag_name == 'div' and 'courseblock' in element.get('class', []):
                        course_code_elem = element.find(class_='coursecode')
                        course_title_elem = element.find(class_='coursetitle')
                        course_attrs_elem = element.find(class_='noindent courseblockattr hours')
                        satisfies_elem = element.find(class_='noindent courseblockattr')

                        # Check if elements are found before accessing their text attributes
                        course_code = course_code_elem.text.strip() if course_code_elem else ''
                        data[heading_text].append(course_code)
                        course_title = course_title_elem.text.strip() if course_title_elem else ''
                        data[heading_text].append(course_title)
                        course_attrs = course_attrs_elem.get_text(" ",strip=True) if course_attrs_elem else ''
                        data[heading_text].append(course_attrs)
                        satisfies = satisfies_elem.get_text(" ",strip=True) if satisfies_elem else ''
                        data[heading_text].append(satisfies)

                    elif tag_name == 'div' and 'cl-menu' in element.get('id', ''):
                        break

                output_file_path = f'{self.url_hash}.txt'

                        # Find all relevant elements
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    for heading, content in data.items():
                        output_file.write(f"{heading}\n")
                        for item in content:
                            if isinstance(item, str):
                                output_file.write(f"{item}\n")
                            elif isinstance(item, list):
                                for row in item:
                                    output_file.write(f"{', '.join(row)}\n")
                        output_file.write(f"Information Source: {self.url}\n\n")

                self.remove_commas_and_save(output_file_path)


            else:
                print(f"Failed to retrieve URL: {self.url}. Status code: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Error while fetching URL: {self.url}. Exception: {e}")

    def remove_commas_and_save(self, input_file):
        try:
            with open(input_file, 'r') as infile:
                content = infile.read()
                content_without_commas = content.replace(',', '')

            with open(input_file, 'w') as outfile:
                outfile.write(content_without_commas)
                
        except FileNotFoundError:
            print(f"Error: File '{input_file}' not found.")

class ScrapeAll:
    def __init__(self, url, url_hash):
        self.url = url
        self.url_hash = url_hash

    def scrape_other(self):
        elements = partition(url=self.url, strategy='auto', new_after_n_chars=1600)
        self.chunks = chunk_by_title(elements)

    def clean_unstructured(self):
        self.chunks_cleaned=[]
        for i in self.chunks:
            #have to check whether cleaning extra_whitespaces can disturb RecursiveCharacterTextSplitter
            # check @ https://unstructured-io.github.io/unstructured/core/cleaning.html
            #is_html() is designed to be used in ScrapeIIT class, have to change
            #to-do: change extra_whitespace to False
            j=clean(str(i), bullets=True, extra_whitespace=True, dashes=True)  
            k = clean_bullets(j)
            # Check if k is empty because clean_ordered_bullets() will throw an error if k is empty
            if not k:  
                l=k
            else:
                l = clean_ordered_bullets(k)
            m = clean_non_ascii_chars(l)
            self.chunks_cleaned.append(m)
            self.chunks_cleaned = list(dict.fromkeys(self.chunks_cleaned)) #remove duplicates

    def concatenate_strings(self):
        self.new_list = []
        for i in range(len(self.chunks_cleaned)):
            if self.chunks_cleaned[i] and self.chunks_cleaned[i][0].isupper():
                self.new_list.append(self.chunks_cleaned[i])
            elif self.new_list:  # Check if new_list is not empty
                self.new_list[-1] += "" + self.chunks_cleaned[i]
        return self.new_list

    #creating standard chunks as per our approach for dealing with IIT chunks in scrape_iit and scrape_bulletin methods
    def standardize_chunks(self):
        self.standard_list = []
        for i in self.new_list:
            j = f"sos: {i} \nInformation Source: {self.url}"
            self.standard_list.append(j)
        return self.standard_list
    
    def store_as_text(self):
        # Write the combined text to the output file
        output_file_path = f'{self.url_hash}.txt'
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for item in self.standard_list:
                output_file.write(f"{item}\n\n")


if __name__ == '__main__':
    scraper = WebScraper(file='/home/ec2-user/ITMT597/chatbot/web_scrapping/combined_urls.csv', use_split=True)
    scraper.scrape()
    
