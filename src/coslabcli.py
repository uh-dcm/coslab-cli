import os
import argparse
import json
import imghdr ## todo: will be depracted

import yaml
from progress.bar import Bar

from coslab import aws
from coslab import googlecloud
from coslab import azure_vision
from coslab import taggerresults
## import imagetaggers

def image_files(folder):
    out = []
    for root, folders, files in os.walk(folder):
        files = map(lambda f: root + '/' + f, files)
        files = filter(lambda f: imghdr.what(f) != None, files)
        out += files
    return list(out)

def load_config(file):
    if file.endswith('.yaml'):
        return yaml.safe_load(open(file))
    if file.endswith('.json'):
        return json.load(open(file))
    return {}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Automatically tag pictures using exernal APIs."
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file",
        required=True
    )

    ## URL interface not yet implemented
    group = parser.add_mutually_exclusive_group(required=True)
    ## group.add_argument("--file", type=str, help="Path to file containing URLs")
    group.add_argument("--folder", type=str,
                       help="Path to folder containing images")
    
    services = {
        'aws' : aws.AWS,
        'google': googlecloud.GoogleCloud,
        'azure': azure_vision.Azure
    }

    parser.add_argument(
        '--api',
        choices=services.keys(),
        nargs='+',
        required=True
    )

    parser.add_argument(
        "--sql",
        type=str,
        help="Where the SQL results are stored.",
    )

    parser.add_argument(
        "--pickle",
        type=str,
        help="Where the pickle results are stored.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    out = taggerresults.TaggerResults()
    use_services = []

    for service in args.api:
        tagger = services[ service ].from_config( config )
        use_services.append( tagger )

    if args.folder:
        directory = args.folder
        images = image_files(directory)

        bar = Bar('Images labelled', max=len(images)*len(use_services))

        for image in images:
            for service in use_services:
                if 'minimal_confidence' in config:
                    service.process_local(out, image, min_confidence = config['minimal_confidence'])
                else:
                    service.process_local(out, image)
                bar.next()

        bar.finish()

    if args.sql:
        out.export_sql( args.sql )

    if args.pickle:
        out.export_pickle( args.pickle )

