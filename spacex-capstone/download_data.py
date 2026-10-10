"""Download primary-source inputs. python download_data.py"""
from pathlib import Path
import urllib.request
DATA=Path(__file__).parent/'data';DATA.mkdir(exist_ok=True)
BASE='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/'
SOURCES={
 'dataset_part_1.csv':BASE+'datasets/dataset_part_1.csv',
 'dataset_part_2.csv':BASE+'datasets/dataset_part_2.csv',
 'spacex_launch_dash.csv':BASE+'datasets/spacex_launch_dash.csv',
 'spacex_launch_geo.csv':BASE+'datasets/spacex_launch_geo.csv',
 'Spacex.csv':BASE+'labs/module_2/data/Spacex.csv',
 'API_call_spacex_api.json':BASE+'datasets/API_call_spacex_api.json',
 'wikipedia_launches.html':'https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches_%282010%E2%80%932019%29',
 'osm_proximity.xml':'https://api.openstreetmap.org/api/0.6/map?bbox=-80.59,28.55,-80.56,28.58'
}
if __name__=='__main__':
 for name,url in SOURCES.items():
  request=urllib.request.Request(url,headers={'User-Agent':'EducationalCapstone/1.0'})
  (DATA/name).write_bytes(urllib.request.urlopen(request,timeout=60).read())
  print(name)
