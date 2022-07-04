from typing import List, Callable, NamedTuple
import streamlit as st
import pandas as pd
from random import randint

class Line:
    def __init__(self, b, m):
        self.b = b
        self.m = m
    
    def get_y(self, x): return self.b + self.m * x

    def __str__(self):
        return f"Line(m={self.m}, b={self.b})"

EPS = 0.01

class LinearRegression:
    class Point(NamedTuple):
        x: float
        y: float
    
    class DY(NamedTuple):
        true_y: float
        calculated_y: float
    
    @staticmethod
    def square_err(data: List[DY]) -> float:
        return sum(
            map(
                lambda x: (x.true_y - x.calculated_y) ** 2, 
                data
            )
        )
    
    def __init__(self, data: List[Point], err_function: Callable[[List[DY]], float]):
        self.data = data
        self.err_function = err_function
    
    def _calculate_line_error(self, m, b):
        line = Line(b, m)
        calculated_y = [line.get_y(point.x) for point in self.data]
        return self.err_function(
                map(
                    lambda index: LinearRegression.DY(self.data[index].y, calculated_y[index]),
                    range(len(self.data))
                )
            )
    
    def _calculate_step(self, curr_err, next_err, value):
        return value + EPS if curr_err - next_err > 0 else value - EPS

    def train(self):
        m = -3
        b = 0
        for _ in range(10000):
            m_err, m_d_err = self._calculate_line_error(m, b), self._calculate_line_error(m+EPS, b)
            b_err, b_d_err = self._calculate_line_error(m, b), self._calculate_line_error(m, b+EPS)
            m = self._calculate_step(m_err, m_d_err, m)
            b = self._calculate_step(b_err, b_d_err, b)
        return Line(b, m), self._calculate_line_error(m, b)
            


data_length = st.slider("Pick length of input data", 3, 100)

data_min_value, data_max_value = st.slider("Pick range of input data", -100, 100, (0, 50))


data: List[LinearRegression.Point] = [
    LinearRegression.Point(i, randint(data_min_value, data_max_value)) for i in range(data_length)
]

regression = LinearRegression(data, LinearRegression.square_err)
line, err = regression.train()

# st.write(
#     "# Hello world"
# )

pd_data = pd.DataFrame(
    [[line.get_y(x), data[x].y] for x in range(len(data))],
    columns=['predict', 'actual']
)

st.line_chart(
    data=pd_data
)