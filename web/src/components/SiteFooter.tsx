export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <p className="site-footer__note">
        界面视觉灵感来自人格类型框架（如{" "}
        <a
          href="https://www.16personalities.com/"
          target="_blank"
          rel="noopener noreferrer"
        >
          16Personalities
        </a>
        ），插画素材存放于项目 <code>Material/</code>{" "}
        目录，与本站无官方关联。
      </p>
      <p className="site-footer__disclaimer">
        免责声明：本工具为基于统计学习的演示模型，结果仅供研究娱乐，不构成心理测评或人格诊断建议。
      </p>
    </footer>
  );
}
