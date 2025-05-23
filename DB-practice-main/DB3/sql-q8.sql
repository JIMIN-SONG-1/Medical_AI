SELECT * FROM emp;
-- 각 부서별 급여를 제일 많이 받는 사원의 월급을 받는 사원들을 조회하는 쿼리를 작성해주세요.
select * from emp
where sal in (select max(sal) from emp group by deptno);