# -*- coding: utf-8 -*-
import scrapy
import re


class PronunSpider(scrapy.Spider):
    name = "ciba"
    start_urls = [
    ]

    # generate url
    try:
        word_list = open('../word_list', 'r')
        for line in word_list.readlines():
            start_urls.append('http://www.iciba.com/' + line.strip())
    except IOError:
        print "Please Check word_list"
    finally:
        word_list.close()

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
