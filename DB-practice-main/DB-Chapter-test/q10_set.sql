-- code_manager
CREATE TABLE code_manager (
  sequence INT NOT NULL,
  status_kind VARCHAR(50) NULL,
  status_code VARCHAR(4) NULL,
  status_name VARCHAR(50) NULL,
  PRIMARY KEY (sequence))
;
INSERT INTO code_manager values(1, 'membership_grade', '0001', '골드');
INSERT INTO code_manager values(2, 'membership_grade', '0002', '실버');
INSERT INTO code_manager values(3, 'membership_grade', '0003', '브론즈');
INSERT INTO code_manager values(4, 'customer_status', '1001', '정상');
INSERT INTO code_manager values(5, 'customer_status', '1002', '휴면');
INSERT INTO code_manager values(6, 'customer_status', '1003', '탈퇴');
INSERT INTO code_manager values(7, 'rider_status', '2001', '근무');
INSERT INTO code_manager values(8, 'rider_status', '2002', '휴가');
INSERT INTO code_manager values(9, 'rider_status', '2003', '퇴직');
INSERT INTO code_manager values(10, 'order_status', '3001', '주문성공');
INSERT INTO code_manager values(11, 'order_status', '3002', '주문취소');
INSERT INTO code_manager values(12, 'order_status', '3003', '기타취소');
INSERT INTO code_manager values(13, 'employee_status', '4001', '근무');
INSERT INTO code_manager values(14, 'employee_status', '4002', '휴가');
INSERT INTO code_manager values(15, 'employee_status', '4003', '퇴직');


-- customer
CREATE TABLE customer (
  customer_id INT NOT NULL,
  phone_number VARCHAR(50) NULL,
  username VARCHAR(50) NULL,
  customer_status VARCHAR(4) NULL,
  membership_grade VARCHAR(4) NULL,
  signup_date DATETIME NULL,
  PRIMARY KEY (customer_id),
  UNIQUE INDEX phone_number_UNIQUE (phone_number ASC)
)
;
INSERT INTO customer values(1, '010-1234-5678', '김다혜', '1001', null, '2022-08-28 14:00:45');
INSERT INTO customer values(2, '010-1685-7542', '이우주', '1001', null, '2022-09-02 18:01:12');
INSERT INTO customer values(3, '010-4572-4235', '이지은', '1001', null, '2022-09-05 09:00:15');
INSERT INTO customer values(4, '010-5721-8872', '김창희', '1001', null, '2022-09-16 14:45:25');
INSERT INTO customer values(5, '010-7215-8723', '김시우', '1001', null, '2022-09-20 07:26:35');
INSERT INTO customer values(6, '010-4659-2156', '박태환', '1001', null, '2022-09-20 04:33:15');
INSERT INTO customer values(7, '010-3896-4523', '강호동', '1001', null, '2022-09-21 15:11:42');
INSERT INTO customer values(8, '010-7542-5423', '이은지', '1001', null, '2022-09-21 16:27:23');
INSERT INTO customer values(9, '010-2453-9634', '김현준', '1001', null, '2022-09-25 18:05:14');
INSERT INTO customer values(10, '010-1557-2422', '박정우', '1001', null, '2022-10-01 22:10:12');
INSERT INTO customer values(11, '010-2234-5578', '김정희', '1001', null, '2022-08-29 16:10:57');
INSERT INTO customer values(12, '010-6586-4452', '김서우', '1001', null, '2022-09-12 19:04:14');
INSERT INTO customer values(13, '010-5826-2457', '김희은', '1001', null, '2022-09-13 23:00:12');
INSERT INTO customer values(14, '010-9215-5712', '김정희', '1001', null, '2022-09-13 16:15:38');
INSERT INTO customer values(15, '010-4882-6681', '김희정', '1001', null, '2022-09-14 17:23:55');
INSERT INTO customer values(16, '010-9658-5243', '박정희', '1001', null, '2022-09-15 22:15:27');
INSERT INTO customer values(17, '010-4575-1279', '문은지', '1001', null, '2022-09-19 09:33:54');
INSERT INTO customer values(19, '010-9874-6612', '이윤희', '1001', null, '2022-09-05 17:02:10');
INSERT INTO customer values(20, '010-5748-8574', '김이준', '1001', null, '2022-09-01 09:55:20');
INSERT INTO customer values(21, '010-8542-4412', '이건우', '1001', null, '2022-09-15 16:12:00');
INSERT INTO customer values(22, '010-2456-7456', '박준혁', '1001', null, '2022-09-16 17:54:30');

-- rider
CREATE TABLE rider (
  rider_id INT NOT NULL,
  phone_number VARCHAR(50) NULL,
  username VARCHAR(50) NULL,
  rider_status VARCHAR(4) NULL,
  superior INT NULL,
  PRIMARY KEY (rider_id),
  UNIQUE INDEX phone_number_UNIQUE (phone_number ASC)
)
;
INSERT INTO rider values(1, '010-2143-1463', '이승우', '2001', null);
INSERT INTO rider values(2, '010-4353-4748', '김승현', '2001', null);
INSERT INTO rider values(3, '010-2375-2476', '박준혁', '2001', null);
INSERT INTO rider values(4, '010-6589-3479', '강지환', '2001', 2);
INSERT INTO rider values(5, '010-8863-1254', '김윤우', '2001', 2);
INSERT INTO rider values(6, '010-2163-2347', '김승민', '2001', null);
INSERT INTO rider values(7, '010-5412-4127', '이윤재', '2002', null);
INSERT INTO rider values(8, '010-5784-6857', '김하람', '2002', 3);
INSERT INTO rider values(9, '010-9684-2472', '김준수', '2001', 3);
INSERT INTO rider values(10, '010-1576-3472', '박도훈', '2003', null);
INSERT INTO rider values(11, '010-6849-9843', '김지완', '2002', null);


-- card_company_manager
CREATE TABLE card_company_manager (
  card_company_id INT NOT NULL,
  company_name VARCHAR(50) NULL,
  fee DECIMAL(7,4) NULL,
  from_date DATETIME NULL,
  to_date DATETIME NULL,
  PRIMARY KEY (card_company_id)
)
;
INSERT INTO card_company_manager VALUES(1, '삼성카드', 1.32, '2022-08-29 00:00:00', '2080-09-14 00:00:00');
INSERT INTO card_company_manager VALUES(2, '현대카드', 1.32, '2022-08-29 00:00:00', '2080-09-14 00:00:00');
INSERT INTO card_company_manager VALUES(3, '신한카드', 3.32, '2022-08-29 00:00:00', '2080-09-14 00:00:00');
INSERT INTO card_company_manager VALUES(4, '롯데카드', 3.35, '2022-08-29 00:00:00', '2080-09-14 00:00:00');


-- orders
CREATE TABLE orders (
  order_number VARCHAR(13) NOT NULL,
  order_status VARCHAR(4) NULL,
  order_date DATETIME NULL,
  cancel_date DATETIME NULL,
  customer INT NULL,
  delivery_rider INT NULL,
  card_company INT NULL,
  PRIMARY KEY (order_number)
)
;
INSERT INTO orders values('20220901A0001', '3001', '2022-08-29 16:10:57', null, 11, 1, 1);
INSERT INTO orders values('20220904A0001', '3001', '2022-09-04 17:57:23', null, 2, 2, 1);
INSERT INTO orders values('20220904A0002', '3001', '2022-09-04 18:10:57', null, 11, 3, 2);
INSERT INTO orders values('20220907A0001', '3001', '2022-09-07 19:07:22', null, 1, 4, 3);
INSERT INTO orders values('20220907A0002', '3001', '2022-09-07 20:11:21', null, 2, 2, 2);
INSERT INTO orders values('20220908A0001', '3001', '2022-09-08 12:05:54', null, 11, 1, 2);
INSERT INTO orders values('20220917A0001', '3001', '2022-09-17 11:10:41', null, 4, 3, 1);
INSERT INTO orders values('20220917A0002', '3001', '2022-09-17 15:07:56', null, 2, 4, 1);
INSERT INTO orders values('20220917A0003', '3001', '2022-09-17 22:39:23', null, 3, 5, 2);
INSERT INTO orders values('20220920A0001', '3001', '2022-09-20 19:10:56', null, 5, 6, 2);

INSERT INTO orders values('20220920A0002', '3001', '2022-09-20 19:27:41', null, 1, 1, 1);
INSERT INTO orders values('20220920A0003', '3001', '2022-09-20 19:34:47', null, 2, 2, 1);
INSERT INTO orders values('20220921A0001', '3001', '2022-09-21 20:00:21', null, 11, 3, 2);
INSERT INTO orders values('20220922A0002', '3001', '2022-09-22 15:50:15', null, 8, 4, 3);
INSERT INTO orders values('20220922A0003', '3001', '2022-09-22 16:55:47', null, 1, 5, 4);
INSERT INTO orders values('20220923A0001', '3001', '2022-09-23 19:13:21', null, 3, 6, 4);
INSERT INTO orders values('20220923A0002', '3001', '2022-09-23 20:00:21', null, 3, 1, 2);
INSERT INTO orders values('20220924A0001', '3001', '2022-09-24 21:48:21', null, 3, 2, 3);
INSERT INTO orders values('20220930A0001', '3001', '2022-09-30 19:19:32', null, 9, 2, 4);
INSERT INTO orders values('20220930A0002', '3001', '2022-09-30 21:17:28', null, 10, 3, 2);

INSERT INTO orders values('20220922A0004', '3002', '2022-09-22 18:49:54', '2022-09-22 19:00:00', 9, 1, 4);
INSERT INTO orders values('20220922A0005', '3002', '2022-09-22 19:16:28', '2022-09-23 09:03:21', 10, 5, 2);
INSERT INTO orders values('20220922A0006', '3001', '2022-09-22 22:37:28', null, 10, 3, 2);
INSERT INTO orders values('20220922A0007', '3003', '2022-09-22 23:00:00', '2022-09-22 23:02:00', 9, null, 4);
INSERT INTO orders values('20220917A0004', '3001', '2022-09-17 22:39:00', null, 2, null, 2);
INSERT INTO orders values('20220905A0001', '3001', '2022-09-05 20:00:00', null, 2, 3, 1);

-- product
CREATE TABLE product (
  product_id INT NOT NULL,
  product_name VARCHAR(200) NULL,
  stock_count INT NULL,
  standard_cost INT NULL,
  PRIMARY KEY (product_id)
)
;
INSERT INTO product values(1, '계란', 30, 3800);
INSERT INTO product values(2, '삼겹살', 50, 1450);
INSERT INTO product values(3, '제육불고기', 20, 6200);
INSERT INTO product values(4, '닭볶음탕', 10, 7800);
INSERT INTO product values(5, '연어', 27, 10900);
INSERT INTO product values(6, '바나나', 55, 6700);
INSERT INTO product values(7, '토마토', 10, 7000);
INSERT INTO product values(8, '샤인머스켓', 3, 14500);
INSERT INTO product values(9, '메론', 4, 12000);
INSERT INTO product values(10, '감귤', 58, 6000);
INSERT INTO product values(11, '감자튀김', 6, 5000);
INSERT INTO product values(12, '해쉬브라운', 11, 5500);
INSERT INTO product values(13, '오징어튀김', 27, 6000);
INSERT INTO product values(14, '치즈볼', 41, 7000);
INSERT INTO product values(15, '만두', 2, 2000);

-- orders_detail
CREATE TABLE orders_detail (
  sequence INT NOT NULL,
  orders VARCHAR(13) NULL,
  product INT NULL,
  order_count INT NULL,
  pay_price INT NULL,
  PRIMARY KEY (sequence)
)
;

INSERT INTO orders_detail values(1, '20220901A0001', 1, 2, 2*3800);
INSERT INTO orders_detail values(2, '20220901A0001', 3, 1, 1*6200);
INSERT INTO orders_detail values(3, '20220901A0001', 4, 1, 1*7800);
INSERT INTO orders_detail values(4, '20220901A0001', 5, 1, 1*10900);

INSERT INTO orders_detail values(5, '20220904A0001', 8, 2, 2*14500);
INSERT INTO orders_detail values(6, '20220904A0001', 15, 1, 1*2000);

INSERT INTO orders_detail values(7, '20220904A0002', 2, 1, 1*1450);
INSERT INTO orders_detail values(8, '20220904A0002', 13, 1, 1*6000);
INSERT INTO orders_detail values(9, '20220904A0002', 15, 1, 1*2000);
INSERT INTO orders_detail values(10, '20220904A0002', 7, 1, 1*7000);

INSERT INTO orders_detail values(11, '20220907A0001', 5, 1, 1*10900);
INSERT INTO orders_detail values(12, '20220907A0001', 8, 2, 2*14500);
INSERT INTO orders_detail values(13, '20220907A0001', 9, 1, 1*12000);

INSERT INTO orders_detail values(14, '20220907A0002', 12, 3, 3*5500);
INSERT INTO orders_detail values(15, '20220907A0002', 13, 3, 3*6000);

INSERT INTO orders_detail values(16, '20220908A0001', 1, 1, 1*3800);
INSERT INTO orders_detail values(17, '20220908A0001', 4, 2, 2*7800);
INSERT INTO orders_detail values(18, '20220908A0001', 5, 2, 2*10900);
INSERT INTO orders_detail values(19, '20220908A0001', 8, 1, 1*14500);
INSERT INTO orders_detail values(20, '20220908A0001', 13, 3, 3*6000);
INSERT INTO orders_detail values(21, '20220908A0001', 14, 3, 3*7000);

INSERT INTO orders_detail values(22, '20220917A0001', 3, 3, 3*6200);

INSERT INTO orders_detail values(23, '20220917A0002', 6, 3, 3*6700);
INSERT INTO orders_detail values(24, '20220917A0002', 7, 2, 2*7000);
INSERT INTO orders_detail values(25, '20220917A0002', 8, 3, 3*14500);

INSERT INTO orders_detail values(26, '20220917A0003', 1, 3, 3*3800);
INSERT INTO orders_detail values(27, '20220917A0003', 5, 2, 2*10900);
INSERT INTO orders_detail values(28, '20220917A0003', 6, 1, 1*6700);
INSERT INTO orders_detail values(29, '20220917A0003', 10, 1, 1*6000);
INSERT INTO orders_detail values(30, '20220917A0003', 8, 1, 1*14500);

INSERT INTO orders_detail values(31, '20220920A0001', 2, 3, 3*1450);
INSERT INTO orders_detail values(32, '20220920A0001', 3, 3, 3*6200);

INSERT INTO orders_detail values(33, '20220920A0002', 5, 2, 2*10900);
INSERT INTO orders_detail values(34, '20220920A0002', 8, 5, 5*14500);
INSERT INTO orders_detail values(35, '20220920A0002', 6, 1, 1*6700);
INSERT INTO orders_detail values(36, '20220920A0002', 10, 1, 1*6000);
INSERT INTO orders_detail values(37, '20220920A0002', 11, 1, 1*5000);

INSERT INTO orders_detail values(38, '20220920A0003', 4, 2, 2*7800);
INSERT INTO orders_detail values(39, '20220920A0003', 13, 2, 2*6000);
INSERT INTO orders_detail values(40, '20220920A0003', 2, 2, 2*1450);

INSERT INTO orders_detail values(41, '20220921A0001', 1, 10, 10*3800);
INSERT INTO orders_detail values(42, '20220921A0001', 2, 5, 5*1450);

INSERT INTO orders_detail values(43, '20220922A0002', 1, 3, 3*3800);
INSERT INTO orders_detail values(44, '20220922A0002', 6, 2, 2*6700);
INSERT INTO orders_detail values(45, '20220922A0002', 8, 1, 1*14500);
INSERT INTO orders_detail values(46, '20220922A0002', 9, 1, 1*12000);
INSERT INTO orders_detail values(47, '20220922A0002', 14, 1, 1*7000);

INSERT INTO orders_detail values(48, '20220922A0003', 3, 3, 3*6200);

INSERT INTO orders_detail values(49, '20220923A0001', 11, 1, 1*5000);
INSERT INTO orders_detail values(50, '20220923A0001', 12, 1, 1*5500);

INSERT INTO orders_detail values(51, '20220923A0002', 4, 1, 1*7800);
INSERT INTO orders_detail values(52, '20220923A0002', 10, 10, 10*6000);

INSERT INTO orders_detail values(53, '20220924A0001', 15, 5, 5*2000);

INSERT INTO orders_detail values(54, '20220930A0001', 5, 5, 5*10900);
INSERT INTO orders_detail values(55, '20220930A0001', 8, 3, 3*14500);

INSERT INTO orders_detail values(56, '20220930A0002', 1, 2, 2*3800);
INSERT INTO orders_detail values(57, '20220930A0002', 6, 5, 5*6700);
INSERT INTO orders_detail values(58, '20220930A0002', 11, 3, 3*5000);

INSERT INTO orders_detail values(59, '20220922A0004', 1, 2, 2*3800);
INSERT INTO orders_detail values(60, '20220922A0004', 2, 2, 2*1450);
INSERT INTO orders_detail values(61, '20220922A0004', 3, 2, 2*6200);

INSERT INTO orders_detail values(62, '20220922A0005', 6, 3, 3*6700);
INSERT INTO orders_detail values(63, '20220922A0005', 7, 3, 3*7000);

INSERT INTO orders_detail values(64, '20220922A0006', 11, 3, 3*5000);
INSERT INTO orders_detail values(65, '20220922A0006', 12, 3, 3*5500);

INSERT INTO orders_detail values(66, '20220922A0007', 12, 3, 3*5500);

INSERT INTO orders_detail values(67, '20220917A0004', 1, 2, 2*3800);
INSERT INTO orders_detail values(68, '20220905A0001', 1, 2, 2*3800);
