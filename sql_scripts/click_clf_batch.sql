with d as (
select row_number() over () as id, matrix.* as matrix
from click_prediction_small_mlpclassifier_scale() as matrix
)
,

test_mat as (
SELECT m.id, u.ord - 1 as ord, u.val
FROM   d m,
       LATERAL unnest(m.matrix) WITH ORDINALITY AS u(val, ord)
)
,

input_weights as
(
SELECT m1.id, m2.col, sum(m1.val * m2.val) AS val
FROM   test_mat m1, nn_matrix_click_prediction_small m2
WHERE  m2.id = 0
AND    m1.ord = m2.row
GROUP BY m1.id, m2.col
ORDER BY m1.id, m2.col
)
,

activation as
(
select iw.id, iw.col, 1/(1 + EXP(-(iw.val + nn.val))) as val
from input_weights iw, nn_matrix_click_prediction_small nn
where nn.id = 2 and iw.col = nn.row
-- group by 1, 2, 3
),

output_weights as
(
select m1.id, sum(m1.val*nn.val) as val
from activation m1, nn_matrix_click_prediction_small nn
where nn.id = 1 and m1.col = nn.row
group by m1.id
)

select m1.id, 1/(1 + EXP(-(m1.val + nn.val))) as val
from output_weights m1, nn_matrix_click_prediction_small nn
where nn.id = 3

----------------

select score_lr_click_prediction_small(s.s) from click_prediction_small_logisticregression_scale() s
