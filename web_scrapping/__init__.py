from langchain.document_loaders import RecursiveUrlLoader
import pandas as pd
from urllib.parse import urlparse
from tqdm import tqdm
import os


class Url_Crawl(RecursiveUrlLoader):
    def __init__(self, base_url, depth):
        super().__init__(url=base_url, max_depth=depth)
        self.base_url = base_url
        self.max_depth = depth
        self.prevent_outside = True # allow crawling outside the base url
        #self.tld_only = True # consider top level domain only

    def get_child_urls(self):
        # Initialize a set to store visited URLs
        visited = set()
        
        # Initialize a list to store the collected URLs
        self.child_urls = []

        # Call the _get_child_links_recursive method to start crawling
        for document in tqdm(self._get_child_links_recursive(self.base_url, visited)):
            self.child_urls.append(document.metadata['source'])  

        return self.child_urls

    def filter_urls(self):
        """ Filter out URLs containing a question mark
        because these urls are not useful for our purpose
        such urls mostly contain search results, css files, etc.
        hence filtering out urls containing a question mark
        """

        self.filtered_urls = (url for url in self.child_urls if '?' not in url)

        return self.filtered_urls    

class MultiCrawler:
    def __init__(self, urls_with_depths):
        # Remove duplicates based on the base URL
        base_urls = {}
        self.all_urls = []

        for url, depth in urls_with_depths:
            parsed = urlparse(url)
            if not parsed.scheme: # if no scheme is specified, add https://
                url = 'https://' + url
            base_url = urlparse(url).scheme + "://" + urlparse(url).netloc + urlparse(url).path
            
            # If the base URL is already in the dictionary and the new depth is greater, update the depth
            if base_url in base_urls and depth > base_urls[base_url]:
                base_urls[base_url] = depth
                print(f'\nduplicate url: {base_url} with depth: {depth}; the one with highest depth level will be preserved.\n')
            elif base_url not in base_urls:
                base_urls[base_url] = depth
        
        self.urls_with_depths = list(base_urls.items())
        print(self.urls_with_depths)
        
    #crawl each url upto the specified depth and get respective child urls
    def crawl(self):
        for url, depth in self.urls_with_depths:
            crawler = Url_Crawl(base_url=url, depth=depth)
            crawler.get_child_urls()
            self.filtered_urls = crawler.filter_urls()
            self.all_urls.extend(self.filtered_urls)
    
    def process_urls(self):
        """"there are some urls especially in bulletin.iit.edu that
        have duplicate content. One in html form and other in pdf form.
        Here we are doing 2 things mainly:
        1. remove the pdf urls with duplicate content
        2. remove the duplicate urls that result after the first step
        3. remove urls with .png, .ico, .svg, .webmanifest extensions"""

        # performing step 1
        processed_urls_1 = (
            url.rsplit('/', 1)[0] if url.endswith('.pdf') and 
            urlparse(url).path.split('/')[-1].replace('.pdf', '') == urlparse(url).path.split('/')[-2] 
            else url 
            for url in self.all_urls
        )
        # performing step 2
        processed_urls_2 = set(url.rstrip('/') for url in processed_urls_1)

        # performing step 3
        unwanted_extensions = ['.png', '.css', '.ico', '.svg', '.webmanifest']
        self.processed_urls_3 = set(url for url in processed_urls_2 if not any(url.endswith(ext) for ext in unwanted_extensions))
        
        return self
    
    # sort the urls in alphabetical order
    def sort_all_urls(self):        
        self.sorted_urls = sorted(self.processed_urls_3, key=lambda x: urlparse(x))
        return self.sorted_urls 
        
    def store_urls(self):
        # export to csv
        # Get the home directory
        home_directory = os.path.expanduser('~')
        # Create the path to the new directory
        new_directory = os.path.join(home_directory, 'chatbot')
        # Create the new directory
        os.makedirs(new_directory, exist_ok=True)
        os.chdir(new_directory)
        pd.DataFrame(self.sorted_urls, columns=['urls']).to_csv('urls_combined.csv', index = False)
            
if __name__ == '__main__':
    crawler = MultiCrawler([('https://www.iit.edu', 2)])   #, ('https://bulletin.iit.edu/', 3) ]) #testing the top level domain url_crawling
    #crawler = MultiCrawler([os.getenv('WEBSITE_URL'), 3])
    crawler.crawl()
    crawler.process_urls()
    crawler.sort_all_urls()
    crawler.store_urls()

    
