-- WITH RECURSIVE 를 이용하여 계층형 질의를 작성해보세요.
with recursive CTE(mentee_id, mento_id, lvl)
AS (
    select mentee_id, mento_id, 0 as lvl
      from MEMBER
      where mento_id is NULL
      UNION ALL
      SELECT a.mentee_id, a.mento_id, b.lvl + 1
      from MEMBER a
      join CTE as b
      ON a.mento_id = b.mentee_id
)

select mentee_id, mento_id, lvl
from CTE
ORDER BY lvl, mentee_id;