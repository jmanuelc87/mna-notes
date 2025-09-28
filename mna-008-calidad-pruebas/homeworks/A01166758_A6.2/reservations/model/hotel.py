from pydantic import BaseModel
from .sqlite3_adapter import execute, select


class Hotel(BaseModel):
    id: int
    name: str
    description: str
    max_floor: int
    max_rooms: int
    max_occupation: int

    def save(self) -> int:
        sql = """INSERT INTO hotels (
            name,
            description,
            max_floor,
            max_rooms,
            max_occupation)
            VALUES(?, ?, ?, ?, ?)"""

        execute(
            sql,
            (
                self.name,
                self.description,
                self.max_floor,
                self.max_rooms,
                self.max_occupation,
            ),
        )

        last = "SELECT max(id) FROM hotels"
        for id in select(last):
            self.id = id[0]
            return id

    def update(self, id: int):
        sql = """UPDATE hotels SET
                name = ?,
                description = ?,
                max_floor = ?,
                max_rooms = ?,
                max_occupation = ? WHERE id = ?"""

        execute(
            sql,
            (
                self.name,
                self.description,
                self.max_floor,
                self.max_rooms,
                self.max_occupation,
                id
            ),
        )

        last = "SELECT max(id) FROM hotels"
        for id in select(last):
            return id

    @staticmethod
    def delete(id: int):
        sql = """DELETE FROM hotels WHERE id = ?"""
        execute(sql, (id,))

    @staticmethod
    def show(id: int):
        sql = """SELECT * from hotels WHERE id = ?"""
        for it in select(sql, (id,)):
            h = Hotel(
                id=it[0],
                name=it[1],
                description=it[2],
                max_floor=it[3],
                max_rooms=it[4],
                max_occupation=it[5],
            )
            return h

    @staticmethod
    def showAll():
        sql = """SELECT * from hotels"""
        result = []
        for it in select(sql):
            h = Hotel(
                id=it[0],
                name=it[1],
                description=it[2],
                max_floor=it[3],
                max_rooms=it[4],
                max_occupation=it[5],
            )
            result.append((it[0], h))
        return result
