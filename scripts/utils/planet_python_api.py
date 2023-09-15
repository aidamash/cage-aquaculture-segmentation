import os
import requests
import asyncio

import math
import rasterio
import matplotlib.pyplot as plt
import planet
from planet import Session

from requests.auth import HTTPBasicAuth

DOWNLOAD_DIR = os.getenv('~/Documents/Modules/Thesis/lake_victoria/data/planet/kiwa/', '.')

def create_requests():
    # The Orders API will be asked to mask, or clip, results to
    # this area of interest.
    aoi1 = {
        "type":
        "Polygon",
        "coordinates": [[[34.030578,-0.7586589999999944],
                         [34.034808,-0.7586589999999944],
                         [34.034808,-0.7565500000000043],
                         [34.030578,-0.7565500000000043],
                         [34.030578,-0.7586589999999944]]]
    }

    # In practice, you will use a Data API search to find items, but
    # for this example take them as given.
    items1 = ['6536032_3638719_2023-05-26_24be']

    order1 = planet.order_request.build_request(
        name="kiwa-order-PSOrthoTile",
        products=[
            planet.order_request.product(item_ids = items1,
                                         product_bundle='analytic_sr_udm2',
                                         item_type='PSOrthoTile')
        ],
        tools=[planet.order_request.clip_tool(aoi=aoi1)])
    # aoi2 = {
    #     "type":
    #     "Polygon",
    #     "coordinates": [[[34.125922,-0.10076200000000313],
    #                      [34.134599,-0.10076200000000313],
    #                      [34.134599,-0.09553799999999057],
    #                      [34.125922,-0.09553799999999057],
    #                      [34.125922,-0.10076200000000313]]]
    # }
    # items2 = ['20221203_070855_76_241f',' 20221203_075142_40_2254']
    # order2 = planet.order_request.build_request(
    # name="utonga-order-PSScene",
    # products=[
    #     planet.order_request.product(item_ids=items2,
    #                                  product_bundle='analytic_sr_udm2',
    #                                  item_type='PSOrthoTile')
    # ],
    # tools=[planet.order_request.clip_tool(aoi=aoi2)])
    return [order1]

async def create_and_download(client, order_detail, directory):
    """Make an order, wait for completion, download files as a single task."""
    with planet.reporting.StateBar(state='creating') as reporter:
        order = await client.create_order(order_detail)
        reporter.update(state='created', order_id=order['id'])
        await client.wait(order['id'], callback=reporter.update_state)

    await client.download_order(order['id'], directory, progress_bar=True)


async def main():
    async with planet.Session() as sess:
        client = sess.client('orders')
        print(client)
        requests = create_requests()

        await asyncio.gather(*[
            create_and_download(client, request, DOWNLOAD_DIR)
            for request in requests
        ])

asyncio.run(main())




