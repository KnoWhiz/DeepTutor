def hello_world():
    """A simple function to test code analysis."""
    return "Hello, World!"

class Calculator:
    """A simple calculator class."""
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
        
    def multiply(self, a, b):
        return a * b
        
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
