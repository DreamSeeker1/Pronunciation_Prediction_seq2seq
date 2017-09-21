import scrapy
import re


class PronunSpider(scrapy.Spider):
    name = "ciba"
    start_urls = [
        'http://www.iciba.com/nice',
        'http://www.iciba.com/flavor'
    ]

    custom_settings = {
        # specifies exported fields and order
        'FEED_EXPORT_FIELDS': ["Word", "American", "English"]
    }

    def parse(self, response):
        Word = re.match('http://www.iciba.com/(.*)', response.url).group(1)
        English = response.css(
            "div.base-speak span:nth-child(1)>span")[0].re('.*\[(.*)\]')
        American = response.css(
            "div.base-speak span:nth-child(2)>span")[0].re('.*\[(.*)\]')

        yield {'English': English[0], 'American': American[0], 'Word': Word}
