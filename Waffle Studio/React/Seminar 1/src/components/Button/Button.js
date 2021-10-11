import "./Button.css"

const Button = ({setAdderSelected}) => {
  const handleButton = () => {
    setAdderSelected("open");
  }

  return <button className="AddStudent" onClick={handleButton}>추가</button>;
}

export default Button;