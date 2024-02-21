import logging
import random
import re
import requests
import time
import json
from datetime import datetime
from requests.adapters import HTTPAdapter
from pprint import pprint
from convert_json_to_block import insertRowIntoBlockTable
logger = logging.getLogger("Base")

class App_Store_Scraper:
    _scheme = "https"

    _landing_host = "apps.apple.com"
    _rss_host = "itunes.apple.com"

    _landing_path = "{country}/app/{app_name}/id{app_id}"
    _rss_path = "{country}/rss/customerreviews/page={page}/id={app_id}/sortby={sort}/json"

    _user_agents = [
        # NOTE: grab from https://bit.ly/2zu0cmU
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322)",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
    ]

    def __init__(
        self,
        country,
        app_name,
        app_id=None,
        log_format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        log_level="INFO",
        log_interval=5,
    ):
        logging.basicConfig(format=log_format, level=log_level.upper())
        self._base_landing_url = f"{self._scheme}://{self._landing_host}"
        self._base_rss_url = f"{self._scheme}://{self._rss_host}"

        self.country = str(country).lower()
        self.app_name = re.sub(r"[\W_]+", "-", str(app_name).lower())
        if app_id is None:
            logger.info("Searching for app id")
            app_id = self.search_id()
        self.app_id = int(app_id)

        self.url = self._landing_url()

        self.reviews = list()
        self.reviews_count = int()

        self._log_interval = float(log_interval)
        self._log_timer = float()

        self._fetched_count = int()

        self._request_offset = 0
        self._request_page = 1
        self._request_sort = "mostRecent"
        
        self._request_headers = {
            "Accept": "application/json",
            "Authorization": self._token(),
            "Connection": "keep-alive",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": self._base_landing_url,
            "Referer": self.url,
            "User-Agent": random.choice(self._user_agents),
        }
        self._request_params = {
            "l": "en-GB",
            "offset": self._request_offset,
            "limit": 5,
            "platform": "web",
            "additionalPlatforms": "appletv,ipad,iphone,mac",
            "filter": ["2", "3", "4", "5"],
            "sort": "leastRecent",
        }
        
        self._response = requests.Response()

        logger.info(
            f"Initialised: {self.__class__.__name__}"
            f"('{self.country}', '{self.app_name}', {self.app_id})"
        )
        logger.info(f"Ready to fetch reviews from: {self.url}")

    def __repr__(self):
        return "{}(country='{}', app_name='{}', app_id={})".format(
            self.__class__.__name__,
            self.country,
            self.app_name,
            self.app_id,
        )

    def __str__(self):
        width = 12
        return (
            f"{'Country'.rjust(width, ' ')} | {self.country}\n"
            f"{'Name'.rjust(width, ' ')} | {self.app_name}\n"
            f"{'ID'.rjust(width, ' ')} | {self.app_id}\n"
            f"{'URL'.rjust(width, ' ')} | {self.url}\n"
            f"{'Review count'.rjust(width, ' ')} | {self.reviews_count}"
        )

    def _landing_url(self):
        landing_url = f"{self._base_landing_url}/{self._landing_path}"
        return landing_url.format(
            country=self.country, app_name=self.app_name, app_id=self.app_id
        )

    def _rss_url(self, pageNum=1):
        rss_url = f"{self._base_rss_url}/{self._rss_path}"
        return rss_url.format(country=self.country, app_id=self.app_id, page=pageNum, sort=self._request_sort)

    def _get(
        self,
        url,
        headers=None,
        params=None,
    ) -> requests.Response:
        with requests.Session() as s:
            s.mount(self._base_rss_url, HTTPAdapter(max_retries=3))
            logger.debug(f"Making a GET request: {url}")
            self._response = s.get(url, headers=headers, params=params)

    def _token(self):
        self._get(self.url)
        tags = self._response.text.splitlines()
        for tag in tags:
            if re.match(r"<meta.+web-experience-app/config/environment", tag):
                token = re.search(r"token%22%3A%22(.+?)%22", tag).group(1)
                return f"bearer {token}"
            
    def _parse_data(self, after, max_rating):
        try:
            response = self._response.json()

            if 'feed' in response and 'entry' in response['feed']:
                for entry in response['feed']['entry']:
                    curr_rating = int(entry['im:rating']['label'])
                    if (curr_rating > max_rating):
                        continue
                    
                    review_data = {
                        'author_name': entry['author']['name']['label'],
                        'author_uri': entry['author']['uri']['label'],
                        'content': entry['content']['label'],
                        'id': entry['id']['label'],
                        'rating': entry['im:rating']['label'],
                        'version': entry['im:version']['label'],
                        'vote_count': entry['im:voteCount']['label'],
                        'vote_sum': entry['im:voteSum']['label'],
                        'title': entry['title']['label'],
                        'updated': entry['updated']['label']
                    }

                    self.reviews.append(review_data)
                    self.reviews_count += 1
                    self._fetched_count += 1
                    logger.debug(f"Fetched {self.reviews_count} review(s)")
            else:
                logger.error(f"Expected 'feed' and 'entry' keys not found in response: {response}")
        except Exception as e:
            logger.error(f"Error parsing data: {e}")
            
    def _log_status(self):
        logger.info(
            f"[id:{self.app_id}] Fetched {self._fetched_count} reviews "
            f"({self.reviews_count} fetched in total)"
        )

    def _heartbeat(self):
        interval = self._log_interval
        if self._log_timer == 0:
            self._log_timer = time.time()
        if time.time() - self._log_timer > interval:
            self._log_status()
            self._log_timer = 0

    def search_id(self):
        search_url = "https://www.google.com/search"
        self._get(search_url, params={"q": f"app store {self.app_name}"})
        pattern = fr"{self._base_landing_url}/[a-z]{{2}}/.+?/id([0-9]+)"
        
        if self._response is not None and self._response.status_code == 200:
            app_id_match = re.search(pattern, self._response.text)
            if app_id_match:
                return app_id_match.group(1)
            else:
                logger.error("No app ID found in the response")
                return None
        else:
            logger.error("Invalid response received")
            return None
        
    def review(self, num_pages=1, after=None, sleep=None, max_rating=5):
        self._log_timer = 0
        if after and not isinstance(after, datetime):
            raise SystemExit("`after` must be a datetime object.")

        page = 1
        
        try:
            while page < num_pages + 1:
                self._heartbeat()
                
                self._get(
                    self._rss_url(pageNum=page),
                )
                
                self._parse_data(after, max_rating)
                page += 1
                if sleep and isinstance(sleep, int):
                    time.sleep(sleep)
        except KeyboardInterrupt:
            logger.error("Keyboard interrupted")
        except Exception as e:
            logger.error(f"Something went wrong: {e}")
        finally:
            self._log_status()
    
    def save_reviews_to_json(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.reviews, file, default=str)
        logger.info(f"Saved reviews to {file_name}")

    def add_to_lens(self, lens_id):
        for block in self.reviews:
            print("block: ", block)
            insertRowIntoBlockTable(block, lens_id)



if __name__ == "__main__":
    pass
    # country = "us"
    # app_name = "tayasui sketches"

    # scraper_instance = App_Store_Scraper(country, app_name)

    # scraper_instance.review(num_pages=10, max_rating=3, after=None, sleep=1)
    
    # pprint(scraper_instance.reviews)
    # pprint(scraper_instance.reviews_count)
    # scraper_instance.save_reviews_to_json('dana_reviews.json')

    # notability, tayasui sketches
