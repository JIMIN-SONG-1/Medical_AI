create table if not exists score(
    name varchar(32) not null,
    korean int,
    english int,
    math int,
    primary key (name)
);

insert into score values('Elice', 100, 80, 60);
insert into score values('Cheshire', 100, 50, 60);
insert into score values('Dodo', 90, 90, 100);
insert into score values('Caterpillar', 75, 85, 100);
insert into score values('Hatter', 90, 80, 90);
insert into score values('Queen of Heart', 80, 75, 80);
insert into score values('Trump', 65, 100, 80);