function greet(name) {
    return `Hello, ${name}!`;
}

class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    introduce() {
        return `I am ${this.name}, ${this.age} years old.`;
    }
    
    haveBirthday() {
        this.age++;
        return `Happy birthday! ${this.name} is now ${this.age} years old.`;
    }
}

module.exports = { greet, Person };
