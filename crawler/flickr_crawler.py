import argparse
import json
import os
import urllib.request
from hashlib import md5
import yaml
from flickrapi import FlickrAPI


class Crawler(object):
    def __init__(self, public_key, secret_key) -> None:
        super().__init__()
        self._service = FlickrAPI(public_key, secret_key, format='parsed-json')
        self._extras = 'url_o, url_c, url_m, url_n'    # for more extra values, see http://librdf.org/flickcurl/api/flickcurl-searching-search-extras.html

    def search(self, keyword, page, per_page):
        # API document: https://stuvel.eu/flickrapi-doc/2-calling.html
        # for more parameters, see https://www.flickr.com/services/api/flickr.photos.search.html
        cats = self._service.photos.search(text=keyword, extras=self._extras,
                                           media='photos', content_type=1,
                                           sort='relevance', page=page, per_page=per_page)
        res = cats['photos']
        return res


def run(keyword, public_key, secret_key, path_to_save_json_dir, path_to_save_image_dir):
    crawler = Crawler(public_key, secret_key)
    start_page = 1
    end_page = 400 + 1
    num_photos_per_page = 10

    for page in range(start_page, end_page):
        print('Page {:d}/{:d}'.format(page, end_page))

        path_to_save_json_file = os.path.join(path_to_save_json_dir, keyword, '{:s}_{:d}.json'.format(keyword, page))
        os.makedirs(os.path.dirname(path_to_save_json_file), exist_ok=True)

        if os.path.exists(path_to_save_json_file):
            with open(path_to_save_json_file, 'r') as f:
                res = json.load(f)
            print('Load JSON from {:s}'.format(path_to_save_json_file))
        else:
            res = crawler.search(keyword, page, num_photos_per_page)
            with open(path_to_save_json_file, 'w') as f:
                json.dump(res, f)
            print('Save JSON to {:s}'.format(path_to_save_json_file))

        # DEBUG
        # from pprint import pprint
        # pprint(res)
        # exit(-1)

        for photo in res['photo']:
            priorities = ['url_c', 'url_m', 'url_o', 'url_n']
            for extra in priorities:
                if extra in photo:
                    url = photo[extra]
                    break
            else:
                print('No image url, skip.')
                continue

            extension = url.split('.')[-1]
            filename = '{:s}.{:s}'.format(md5(url.encode()).hexdigest(), extension)
            path_to_save_image_file = os.path.join(path_to_save_image_dir, keyword, filename)
            os.makedirs(os.path.dirname(path_to_save_image_file), exist_ok=True)

            if os.path.exists(path_to_save_image_file):
                print('Image has already existed, skip')
                continue

            print('Retrieving {:s}...'.format(url))
            urllib.request.urlretrieve(url, path_to_save_image_file)
            print('Save image to {:s}'.format(path_to_save_image_file))


if __name__ == '__main__':
    def main(args):
        keyword = args.keyword
        path_to_save_json_dir = args.json_dir
        path_to_save_image_dir = args.image_dir

        with open('env.yaml', 'r') as f:
            env = yaml.load(f)
            public_key = env['FLICKR_PUBLIC_KEY']
            secret_key = env['FLICKR_SECRET_KEY']

        run(keyword, public_key, secret_key, path_to_save_json_dir, path_to_save_image_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('keyword', type=str)
    parser.add_argument('-jd', '--json_dir', default='./json')
    parser.add_argument('-id', '--image_dir', default='./images')
    main(parser.parse_args())
