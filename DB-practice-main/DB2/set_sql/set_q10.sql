create table if not exists product(
    id int not null,
    name varchar(32) not null,
    stock int,
    expiration_date date,
    list_price int,
    selling_price int,
    primary key(id)
);