-- 1. 경민이보다 중간고사 수학점수를 높거나 같게 받은 학생들을 조회해 주세요.
select * 
from middle_test as m
join students as s
on s.student_id = m.student_id
where m.math >= (
    select math
    from middle_test
    where student_id = '10504'
);