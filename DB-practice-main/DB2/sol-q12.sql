select
    max(price),
    min(price)
from
    books
where
    publication_year >= 2001
    and publication_year <= 2003;