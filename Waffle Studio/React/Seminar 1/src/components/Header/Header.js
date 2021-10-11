import logo from "./logo.svg";
import "./Header.css";

const Header = () => {
  return (
    <div className="Header">
      <a href="https://wafflestudio.com" target="_blank" rel="noreferrer">
        <img src={logo} alt="Waffle Studio" height="100%"/>
      </a>
      <h1 id="Title">와플고등학교 명단 관리 프로그램</h1>
    </div>
  );
}

export default Header;