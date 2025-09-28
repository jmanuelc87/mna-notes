from reservations.model.customer import Customer


def test_hotel_save():
    h = Customer(
        id=0,
        first_name="Juan",
        last_name="Carballo",
    )
    id = h.save()
    res = Customer.show(id[0])
    assert res is not None


def test_hotel_update():
    h = Customer(
        id=0,
        first_name="Juan",
        last_name="Carballo",
    )
    id = h.save()
    h.first_name = "Juan Manuel"
    id = h.update(id[0])
    res = Customer.show(id[0])
    assert res.first_name == h.first_name


def test_hotel_delete():
    all = Customer.showAll()
    for id, h in all:
        Customer.delete(id)
    all = Customer.showAll()
    assert len(all) == 0


def test_hotel_show():
    h = Customer(
        id=0,
        first_name="Juan",
        last_name="Carballo",
    )
    id = h.save()
    res = Customer.show(id[0])
    assert res.first_name == h.first_name
