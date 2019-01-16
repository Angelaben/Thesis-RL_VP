sudo docker stop benj_dock
sudo docker rm benj_dock
sudo docker rmi benj_dock
sudo docker build -t benj_dock .
sudo docker run -dit -v $(pwd)/:/code -p 6767:6767 --name benj_dock benj_dock
# docker exec -it benj_dock bash
# sudo docker cp  benj_dock:/code/shared_folder


python3 -m pip install ipykernel
python3 -m ipykernel install --user