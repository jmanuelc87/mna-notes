from fastapi import FastAPI
from model import hotel
from model import sqlite3_adapter


sqlite3_adapter.start_db()

app = FastAPI()


@app.post("/hotels")
def create_hotel(item: hotel.Hotel):
    id = item.save()
    return id


@app.delete("/hotels/{id}")
def delete_hotel(id: int):
    hotel.Hotel.delete(id)
    return id


@app.get("/hotels/{id}")
def show_hotel(id: int):
    return hotel.Hotel.show(id)


@app.put("/hotels/{id}")
def update_hotel(id: int, item: hotel.Hotel):
    id = item.update(id)
    return id
