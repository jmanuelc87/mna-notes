from reservations.model.hotel import Hotel


def test_hotel_save():
    h = Hotel(
        id=0,
        name="Holiday Inn",
        description="",
        max_floor=1,
        max_rooms=3,
        max_occupation=10
    )
    id = h.save()
    res = Hotel.show(id[0])
    assert res is not None


def test_hotel_update():
    h = Hotel(
        id=0,
        name="Express Inn",
        description="",
        max_floor=1,
        max_rooms=3,
        max_occupation=10
    )

    id = h.save()

    h.name = "Holiday Inn Express"

    id = h.update(id[0])

    res = Hotel.show(id[0])

    assert res.name == h.name


def test_hotel_delete():
    all = Hotel.showAll()

    for id, h in all:
        Hotel.delete(id)

    all = Hotel.showAll()
    assert len(all) == 0


def test_hotel_show():
    h = Hotel(
        id=0,
        name="Express Inn",
        description="",
        max_floor=1,
        max_rooms=3,
        max_occupation=10
    )
    id = h.save()
    res = Hotel.show(id[0])
    assert res.name == h.name
